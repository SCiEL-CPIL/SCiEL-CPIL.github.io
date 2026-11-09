import random
import time
import math
import sys

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # Safety margin

def _neighbours_list(point):
    """Return list of 6 face-neighbours of point in 3D."""
    x, y, z = point
    return [(x+1,y,z), (x-1,y,z), (x,y+1,z), (x,y-1,z), (x,y,z+1), (x,y,z-1)]

def _energy(sequence, chain):
    """Compute total energy: negative count of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours_list(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _build_contact_map(sequence, chain):
    """
    Builds a map of occupied positions and a contact count.
    Returns: occupied (dict pos->idx), contacts (int), h_positions (set of idx where seq is H)
    """
    occupied = {p: i for i, p in enumerate(chain)}
    h_positions = set(i for i, c in enumerate(sequence) if c == 'H')
    contacts = 0
    for i in h_positions:
        p = chain[i]
        for q in _neighbours_list(p):
            j = occupied.get(q)
            if j is not None and j > i + 1:
                contacts += 1
    return occupied, contacts

def _delta_energy(sequence, chain, idx, old_pos, new_pos, occupied, contacts):
    """
    Calculate the change in energy if residue at idx moves from old_pos to new_pos.
    Assumes new_pos is empty and valid.
    """
    # Remove contacts involving idx at old_pos
    # Add contacts involving idx at new_pos
    # Also, moving idx might break/create contacts between its neighbors?
    # No, only contacts involving idx change, UNLESS the move changes the adjacency of idx's neighbors?
    # Wait, the chain connectivity is fixed. The neighbors of idx in the chain (idx-1, idx+1) do not move.
    # So only contacts involving idx itself change.
    
    # However, if idx moves, does it affect contacts between other residues?
    # No, other residues stay put.
    
    delta = 0
    is_h = sequence[idx] == 'H'
    
    if is_h:
        # Check old position: remove contacts with H's at distance > 1
        for q in _neighbours_list(old_pos):
            j = occupied.get(q)
            if j is not None and j != idx and abs(j - idx) > 1:
                if sequence[j] == 'H':
                    delta += 1  # We are removing a contact, so energy increases (becomes less negative)
        
        # Check new position: add contacts with H's at distance > 1
        for q in _neighbours_list(new_pos):
            j = occupied.get(q)
            if j is not None and j != idx and abs(j - idx) > 1:
                if sequence[j] == 'H':
                    delta -= 1  # We are adding a contact, so energy decreases
    
    # Note: The 'occupied' dict must be updated by the caller after accepting the move.
    # This function assumes 'occupied' reflects the state BEFORE the move.
    # We must ensure we don't count idx itself.
    
    return delta

def _greedy_init(sequence, rng):
    """
    Fast greedy initialization with randomization.
    """
    n = len(sequence)
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        neighbors = _neighbours_list(last)
        rng.shuffle(neighbors)
        
        # Filter out occupied
        valid = [p for p in neighbors if p not in occupied]
        if not valid:
            # Dead end, try to backtrack? For init, just fail or restart
            return None
        
        # Score: prefer H neighbors
        best_score = -1
        best_pos = valid[0]
        for p in valid:
            score = 0
            for q in _neighbours_list(p):
                if q in occupied:
                    j = occupied[q]
                    if sequence[j] == 'H' and abs(j - i) > 1:
                        score += 1
            # Add randomness
            score += rng.random() * 0.5
            if score > best_score:
                best_score = score
                best_pos = p
        
        chain.append(best_pos)
        occupied[best_pos] = i
        
    return chain

def _local_search(sequence, chain, time_budget, rng):
    """
    Simulated Annealing with fast delta energy.
    """
    n = len(chain)
    start_time = time.time()
    
    current_chain = chain
    occupied, contacts = _build_contact_map(sequence, current_chain)
    current_energy = -contacts
    best_chain = current_chain
    best_energy = current_energy
    
    T = max(1.0, abs(best_energy) * 0.5)
    cooling = 0.999
    
    while time.time() < start_time + time_budget:
        # Choose move
        if rng.random() < 0.5:
            move = _slide_move(sequence, current_chain, occupied, rng)
        else:
            move = _pivot_move(sequence, current_chain, occupied, rng)
            
        if move is None:
            continue
            
        idx, old_pos, new_pos, delta = move
        
        # Metropolis
        if delta < 0 or rng.random() < math.exp(-delta / T):
            # Accept
            current_chain[idx] = new_pos
            # Update occupied
            del occupied[old_pos]
            occupied[new_pos] = idx
            current_energy += delta
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain[:]
        
        T *= cooling
    
    return best_chain, best_energy

def _slide_move(sequence, chain, occupied, rng):
    """
    Slide move: move a single residue to an adjacent empty site.
    Returns (idx, old_pos, new_pos, delta) or None.
    """
    n = len(chain)
    if n < 3:
        return None
    
    idx = rng.randint(1, n - 2)
    pos = chain[idx]
    neighbors = _neighbours_list(pos)
    rng.shuffle(neighbors)
    
    for new_pos in neighbors:
        if new_pos in occupied:
            continue
        
        # Check if move is valid (unit step to prev/next is maintained by definition of slide? 
        # No, slide moves the node, so distance to prev/next must be 1.
        # In a cubic lattice, if we move a node to an adjacent site, 
        # the distance to its chain neighbors might change.
        # We must check that new_pos is adjacent to chain[idx-1] and chain[idx+1].
        
        prev_pos = chain[idx - 1]
        next_pos = chain[idx + 1]
        
        d1 = abs(prev_pos[0]-new_pos[0]) + abs(prev_pos[1]-new_pos[1]) + abs(prev_pos[2]-new_pos[2])
        d2 = abs(next_pos[0]-new_pos[0]) + abs(next_pos[1]-new_pos[1]) + abs(next_pos[2]-new_pos[2])
        
        if d1 != 1 or d2 != 1:
            continue
        
        # Calculate delta
        delta = _delta_energy(sequence, chain, idx, pos, new_pos, occupied, 0)
        
        return (idx, pos, new_pos, delta)
    
    return None

def _pivot_move(sequence, chain, occupied, rng):
    """
    Pivot move: rotate a suffix.
    This is harder to compute delta for fast because many nodes move.
    For speed, we will only use Slide in the inner SA loop.
    Pivot is used in Perturbation.
    """
    return None # Disable in inner loop for speed

def _perturb(sequence, chain, rng):
    """
    3-Fragment Perturbation.
    Cut chain into 3 parts: A, B, C.
    Keep A and C fixed.
    Rotate/Translate B.
    """
    n = len(chain)
    if n < 10:
        return chain
    
    # Choose cut points
    i = rng.randint(1, n - 2)
    j = rng.randint(i + 1, n - 1)
    
    A = chain[:i]
    B = chain[i:j]
    C = chain[j:]
    
    # We need to reconnect A's end to B's start and B's end to C's start.
    # This is complex. Simpler: Just rotate B around A's last point?
    # If we rotate B, its start point moves. We must ensure the new start is adjacent to A[-1].
    # This is a "Fragment Rotation".
    
    pivot = A[-1]
    axis = rng.randint(0, 2)
    direction = rng.choice([1, -1])
    
    def rotate(point, axis, direction):
        x, y, z = point
        if axis == 0:
            if direction == 1:
                return (x, -z, y)
            else:
                return (x, z, -y)
        elif axis == 1:
            if direction == 1:
                return (z, y, -x)
            else:
                return (-z, y, x)
        else:
            if direction == 1:
                return (y, -x, z)
            else:
                return (-y, x, z)
    
    new_B = []
    for p in B:
        tx = p[0] - pivot[0]
        ty = p[1] - pivot[1]
        tz = p[2] - pivot[2]
        rx, ry, rz = rotate((tx, ty, tz), axis, direction)
        new_B.append((rx + pivot[0], ry + pivot[1], rz + pivot[2]))
    
    new_chain = A + new_B + C
    
    # Check self-avoidance
    if len(set(new_chain)) != n:
        return None
    
    # Check connectivity
    for k in range(n - 1):
        p1 = new_chain[k]
        p2 = new_chain[k+1]
        dist = abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[2]-p2[2])
        if dist != 1:
            return None
            
    return new_chain

def fold(sequence, dimension):
    """
    ILS with Fast Local Search.
    """
    n = len(sequence)
    elapsed = time.time() - _START_TIME
    remaining = _DEADLINE - elapsed
    if remaining < 2.0:
        return [[i, 0, 0] for i in range(n)]
    
    rng = random.Random(54321 + len(sequence))
    
    best_chain = None
    best_energy = float('inf')
    
    # 1. Initialize multiple diverse folds
    inits = []
    for _ in range(5):
        c = _greedy_init(sequence, rng)
        if c:
            e = _energy(sequence, c)
            inits.append((c, e))
            
    if not inits:
        return [[i, 0, 0] for i in range(n)]
        
    inits.sort(key=lambda x: x[1])
    best_chain, best_energy = inits[0]
    
    # 2. ILS Loop
    # Time per iteration
    # Total time budget: remaining - 5s
    total_budget = remaining - 5.0
    if total_budget < 1.0:
        total_budget = 0.5
        
    # We will run local search on the best chain, then perturb, then local search again.
    current_chain = best_chain
    current_energy = best_energy
    
    # Split time: 50% for initial refinement, 50% for ILS
    init_ls_time = total_budget * 0.3
    ils_time = total_budget * 0.7
    
    # Refine initial best
    if init_ls_time > 1.0:
        current_chain, current_energy = _local_search(sequence, current_chain, init_ls_time, rng)
        if current_energy < best_energy:
            best_energy = current_energy
            best_chain = current_chain
    
    # ILS
    if ils_time > 1.0:
        while time.time() < _START_TIME + _DEADLINE - 2.0:
            # Perturb
            new_chain = _perturb(sequence, current_chain, rng)
            if new_chain is None:
                # Try a smaller perturbation or restart
                # For now, just continue
                continue
                
            # Local search on perturbed chain
            # Allocate small time budget per LS
            ls_budget = min(5.0, (ils_time - (time.time() - (_START_TIME + init_ls_time))) / 10.0)
            if ls_budget < 0.5:
                ls_budget = 0.5
                
            refined_chain, refined_energy = _local_search(sequence, new_chain, ls_budget, rng)
            
            if refined_energy < best_energy:
                best_energy = refined_energy
                best_chain = refined_chain
                current_chain = refined_chain
            else:
                # Accept worse with probability? Or just reject?
                # In ILS, we usually accept if it's not too bad, or use SA acceptance.
                # Let's use a simple acceptance: accept if better, or with prob 0.1 if worse.
                if rng.random() < 0.1:
                    current_chain = refined_chain
                    current_energy = refined_energy
                else:
                    # Keep current_chain as is for next perturbation? 
                    # Usually we perturb the *best* or *current*. 
                    # Let's perturb the *current* (which might be worse than best) to explore.
                    pass
    
    # Final check
    if len(set(best_chain)) != n:
        return [[i, 0, 0] for i in range(n)]
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
