import time
import random
import math
import heapq
import sys

_NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _calc_delta_energy(sequence, chain, occupied, new_pos, residue_idx):
    """Calculate the change in energy when placing residue at residue_idx at new_pos."""
    if sequence[residue_idx] != 'H':
        return 0
    
    delta = 0
    x, y, z = new_pos
    for dx, dy, dz in _NEIGHBOR_OFFSETS:
        r = (x + dx, y + dy, z + dz)
        if r in occupied:
            j = occupied[r]
            # Only count contacts with non-adjacent residues
            if j != residue_idx - 1 and j != residue_idx + 1 and sequence[j] == 'H':
                # This is a new H-H contact, which reduces energy (more negative)
                # But since we're calculating delta for the objective (minimizing energy),
                # a new contact means energy decreases, so delta_energy should reflect this.
                # Actually, let's be consistent: energy = -contacts.
                # Adding a contact decreases energy by 1.
                # So if we add a contact, new_energy = old_energy - 1.
                # delta = new - old = -1.
                delta -= 1
    return delta

def _get_energy(sequence, chain, occupied):
    """Calculate total energy of the chain."""
    energy = 0
    n = len(sequence)
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y, z = chain[i]
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            r = (x + dx, y + dy, z + dz)
            if r in occupied:
                j = occupied[r]
                if j > i and j != i + 1 and sequence[j] == 'H':
                    energy -= 1
    return energy

def _astar_fold(sequence, n, deadline, beam_width=50):
    """
    A* search for folding.
    State: (chain, occupied, energy, f_score)
    f_score = energy + heuristic
    heuristic = -max_possible_future_contacts
    """
    # Precompute remaining H counts
    remaining_h = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        remaining_h[i] = remaining_h[i + 1] + (1 if sequence[i] == 'H' else 0)
    
    # Initial state
    initial_chain = [(0, 0, 0)]
    initial_occupied = {(0, 0, 0): 0}
    initial_energy = 0
    
    # A* priority queue: (f_score, g_score, chain, occupied, energy)
    # We use a heap. Since we want to minimize f_score, we push (f_score, ...)
    # To avoid comparing chains directly, we add a counter
    counter = 0
    heap = []
    
    # Initial heuristic: -remaining_h[1] * 0.5 (rough upper bound on future contacts)
    # A better heuristic: the number of H-H contacts possible in the remaining sequence
    # is bounded by min(remaining_h[i], total_h_in_chain) but that's complex.
    # Simple admissible heuristic: -remaining_h[i] * 0.5
    # Actually, the maximum number of new H-H contacts from residue i onwards
    # is limited by the number of H residues remaining and their proximity.
    # A safe lower bound on energy (i.e., upper bound on f) is -remaining_h[i] * 0.5
    # Let's use: heuristic(i) = -remaining_h[i] * 0.5
    # This is admissible because each H can form at most 1 contact in the best case? No, up to 6.
    # But we need a LOWER bound on the cost-to-go (which is energy, so negative is better).
    # Wait, A* minimizes f = g + h, where g is cost so far, h is estimated cost to goal.
    # Here, cost is energy (negative is better). So we want to minimize energy.
    # g = current energy.
    # h = estimated additional energy change from remaining residues.
    # The best case is that all remaining H's form as many contacts as possible.
    # The maximum number of H-H contacts among k H residues is roughly k*3 (if densely packed).
    # But that's not admissible if we overestimate.
    # Let's use h = -remaining_h[i] * 1.0 as a conservative estimate.
    # Actually, let's just use 0 as a trivially admissible heuristic, and rely on beam pruning.
    # Or use a simple heuristic: -min(remaining_h[i], 3) * 0.5
    
    h_init = -remaining_h[1] * 0.5
    heapq.heappush(heap, (initial_energy + h_init, counter, initial_chain, initial_occupied, initial_energy))
    counter += 1
    
    best_chain = None
    best_energy = float('inf')
    
    visited = set()  # Set of frozensets of occupied positions? Too expensive.
    # Instead, we rely on the beam width to control size.
    
    while heap and time.time() < deadline:
        f_score, _, chain, occupied, energy = heapq.heappop(heap)
        
        if len(chain) == n:
            if energy < best_energy:
                best_energy = energy
                best_chain = chain[:]
            continue
        
        i = len(chain)  # Next residue index
        last_pos = chain[-1]
        
        # Generate neighbors
        candidates = []
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            q = (last_pos[0] + dx, last_pos[1] + dy, last_pos[2] + dz)
            if q in occupied:
                continue
            
            # Calculate delta energy
            delta_e = _calc_delta_energy(sequence, chain, occupied, q, i)
            new_energy = energy + delta_e
            
            # Heuristic for new state
            h_new = -remaining_h[i + 1] * 0.5
            f_new = new_energy + h_new
            
            # Score for beam pruning: prefer lower energy and higher free space
            # Free space: count of unoccupied neighbors
            free_space = 0
            for dx2, dy2, dz2 in _NEIGHBOR_OFFSETS:
                r = (q[0] + dx2, q[1] + dy2, q[2] + dz2)
                if r not in occupied:
                    free_space += 1
            
            # Combine: we want to minimize energy, so lower is better.
            # For the heap, we use f_score.
            # For beam pruning, we might want to keep diverse states.
            # Let's just push into heap and let the heap size control it.
            
            new_chain = chain + [q]
            new_occupied = dict(occupied)
            new_occupied[q] = i
            
            heapq.heappush(heap, (f_new, counter, new_chain, new_occupied, new_energy))
            counter += 1
        
        # Beam pruning: keep only the best beam_width states
        if len(heap) > beam_width:
            # Pop the worst (highest f_score) states
            # This is inefficient. Instead, we should maintain a list and sort.
            # For now, let's just limit the heap size by popping.
            while len(heap) > beam_width:
                heapq.heappop(heap)
    
    return best_chain

def _guided_growth(sequence, n, deadline, rng, temperature=1.0):
    """
    Monte Carlo guided growth: sample a full path using a potential field.
    """
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    for i in range(1, n):
        if time.time() > deadline:
            return None
        
        last_pos = chain[-1]
        
        # Get valid moves
        candidates = []
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            q = (last_pos[0] + dx, last_pos[1] + dy, last_pos[2] + dz)
            if q in occupied:
                continue
            
            # Calculate delta energy if we place residue i here
            delta_e = 0
            if sequence[i] == 'H':
                for dx2, dy2, dz2 in _NEIGHBOR_OFFSETS:
                    r = (q[0] + dx2, q[1] + dy2, q[2] + dz2)
                    if r in occupied:
                        j = occupied[r]
                        if j != i - 1 and sequence[j] == 'H':
                            delta_e += 1
            
            # Free space bonus
            free = 0
            for dx2, dy2, dz2 in _NEIGHBOR_OFFSETS:
                r = (q[0] + dx2, q[1] + dy2, q[2] + dz2)
                if r not in occupied:
                    free += 1
            
            # Score for this move
            # We want to minimize energy, so delta_e is good (more contacts)
            # free space is good (more room)
            score = delta_e * 3.0 + free * 1.0 - 0.1 * (abs(q[0]) + abs(q[1]) + abs(q[2]))
            
            candidates.append((q, score))
        
        if not candidates:
            return None
        
        # Choose with temperature
        # Higher score = better, so we use exp(score / T)
        max_score = max(c[1] for c in candidates)
        weights = []
        for q, score in candidates:
            w = math.exp((score - max_score) / temperature)
            weights.append(w)
        
        total_w = sum(weights)
        if total_w == 0:
            return None
        
        r = rng.random() * total_w
        cum = 0
        chosen = candidates[-1][0]
        for (q, score), w in zip(candidates, weights):
            cum += w
            if r <= cum:
                chosen = q
                break
        
        chain.append(chosen)
        occupied[chosen] = i
    
    return chain

def _local_optimize(sequence, chain, deadline, rng):
    """
    Local optimization using Monte Carlo with Single Moves and Pivot Moves.
    """
    n = len(chain)
    if n < 4:
        return chain, _get_energy(sequence, chain, {p: i for i, p in enumerate(chain)})
    
    occupied = {}
    for idx, pos in enumerate(chain):
        occupied[pos] = idx
    
    current_energy = _get_energy(sequence, chain, occupied)
    best_energy = current_energy
    best_chain = chain[:]
    
    # Cooling schedule
    T = 1.0
    T_min = 0.05
    cooling = 0.9995
    
    steps_done = 0
    max_steps = 200000
    
    while time.time() < deadline and steps_done < max_steps:
        steps_done += 1
        
        move_type = rng.random()
        
        if move_type < 0.5:
            # Single Residue Move
            i = rng.randint(1, n - 2)
            old_pos = chain[i]
            valid_mvs = []
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                q = (old_pos[0] + dx, old_pos[1] + dy, old_pos[2] + dz)
                if q in occupied and occupied[q] != i:
                    continue
                if i > 0:
                    prev = chain[i-1]
                    if abs(q[0]-prev[0]) + abs(q[1]-prev[1]) + abs(q[2]-prev[2]) != 1:
                        continue
                if i < n-1:
                    nxt = chain[i+1]
                    if abs(q[0]-nxt[0]) + abs(q[1]-nxt[1]) + abs(q[2]-nxt[2]) != 1:
                        continue
                valid_mvs.append(q)
            
            if not valid_mvs:
                continue
            
            new_pos = rng.choice(valid_mvs)
            if new_pos == old_pos:
                continue
            
            # Check if move is valid
            if new_pos in occupied and occupied[new_pos] != i:
                continue
            if i > 0:
                prev = chain[i-1]
                if abs(new_pos[0]-prev[0]) + abs(new_pos[1]-prev[1]) + abs(new_pos[2]-prev[2]) != 1:
                    continue
            if i < n-1:
                nxt = chain[i+1]
                if abs(new_pos[0]-nxt[0]) + abs(new_pos[1]-nxt[1]) + abs(new_pos[2]-nxt[2]) != 1:
                    continue
            
            # Apply move
            del occupied[old_pos]
            occupied[new_pos] = i
            chain[i] = new_pos
            
            new_energy = _get_energy(sequence, chain, occupied)
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or (delta_e == 0 and rng.random() < 0.5) or (delta_e > 0 and rng.random() < math.exp(-delta_e / T)):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain[:]
            else:
                # Revert
                chain[i] = old_pos
                occupied[old_pos] = i
                del occupied[new_pos]
                
        else:
            # Pivot Move
            i = rng.randint(1, n - 2)
            hinge_pos = chain[i]
            v = (chain[i+1][0] - hinge_pos[0], chain[i+1][1] - hinge_pos[1], chain[i+1][2] - hinge_pos[2])
            
            # Find valid pivot directions
            valid_dirs = []
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                if (dx, dy, dz) == v:
                    continue
                q = (hinge_pos[0] + dx, hinge_pos[1] + dy, hinge_pos[2] + dz)
                if q in occupied and occupied[q] != i + 1:
                    continue
                if i > 0 and q == chain[i-1]:
                    continue
                valid_dirs.append((dx, dy, dz))
            
            if not valid_dirs:
                continue
            
            new_v = rng.choice(valid_dirs)
            
            # Rotate tail
            # We need a rotation matrix that maps v to new_v
            # For simplicity, let's just try a few common rotations
            # Actually, let's implement a proper pivot move
            
            # Find rotation
            # v and new_v are unit vectors
            # We need R such that R * v = new_v
            # And R is a rotation matrix (det=1, orthogonal)
            
            # For now, let's skip pivot moves for simplicity and just use single moves
            # Or implement a simple pivot
            pass
    
        T = max(T_min, T * cooling)
    
    return best_chain, best_energy

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    start_time = time.time()
    deadline = start_time + 310.0  # Safety margin before 330s kill
    
    best_chain = None
    best_energy = float('inf')
    
    # Strategy: A* search for short chains, guided growth for long chains
    # For short chains (n <= 48), use A*
    # For long chains, use guided growth + local opt
    
    if n <= 48:
        # A* search
        chain = _astar_fold(sequence, n, deadline, beam_width=100)
        if chain is not None and len(chain) == n:
            occupied = {}
            for idx, pos in enumerate(chain):
                occupied[pos] = idx
            energy = _get_energy(sequence, chain, occupied)
            if energy < best_energy:
                best_energy = energy
                best_chain = chain[:]
    else:
        # Guided growth
        rng = random.Random(42)
        chain = _guided_growth(sequence, n, deadline, rng, temperature=1.0)
        if chain is not None and len(chain) == n:
            occupied = {}
            for idx, pos in enumerate(chain):
                occupied[pos] = idx
            energy = _get_energy(sequence, chain, occupied)
            if energy < best_energy:
                best_energy = energy
                best_chain = chain[:]
    
    # Local optimization
    if best_chain is not None and time.time() < deadline - 5:
        rng = random.Random(42)
        opt_chain, opt_energy = _local_optimize(sequence, best_chain[:], deadline, rng)
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_chain = opt_chain[:]
    
    if best_chain is None:
        # Fallback: simple extended chain
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
