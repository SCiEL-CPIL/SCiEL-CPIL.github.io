# EVOLVE-BLOCK-START
import random
import time
import sys
import math
import heapq
from collections import defaultdict

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _beam_search_fold(sequence, dimension, beam_width, deadline, rng):
    """
    Beam search to generate a good initial fold.
    Returns a list of points (tuples).
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # State: (negative_score, chain_tuple, occupied_set, last_pos)
    # We want to maximize score, so we use min-heap with -score
    # To save memory, we only keep the chain and occupied set for the top beams.
    # Actually, keeping occupied set for each beam is expensive.
    # Alternative: keep the chain, and re-compute occupied set if needed? No, too slow.
    # Let's keep the chain as a list of tuples, and a set of occupied sites.
    
    # Initial state
    start_pos = (0, 0, 0)
    initial_chain = [start_pos]
    initial_occupied = {start_pos}
    
    # Heap entries: (-score, tie_breaker, chain, occupied)
    # We use a counter for tie-breaking to avoid comparing chains
    counter = 0
    heap = []
    heapq.heappush(heap, (0, counter, initial_chain, initial_occupied))
    counter += 1
    
    for step in range(1, n):
        if not heap:
            break
        if time.time() > deadline - 1.0:
            break
            
        new_heap = []
        candidates = []
        
        # Expand all beams in the current level
        for score, _, chain, occupied in heap:
            last = chain[-1]
            res = sequence[step]
            
            for nb in _neighbours(last):
                if nb in occupied:
                    continue
                
                new_occupied = occupied | {nb}
                new_chain = chain + [nb]
                
                # Calculate new score (number of H-H contacts)
                new_score = 0
                if res == 'H':
                    for q in _neighbours(nb):
                        if q in new_occupied and q != last:
                            # Check if the other end is H
                            # We need to know the index of q in the chain
                            # Since we are building the chain sequentially, q is in chain[:-1] (excluding last which is not in new_occupied yet? No, last is in occupied)
                            # Wait, q is in new_occupied.
                            # If q == last, it's a bond, not a contact. We excluded q == last.
                            # So q is a previous residue.
                            # We need to find its index.
                            # This is O(N) per contact.
                            # Optimization: keep a map of pos -> index?
                            # That's expensive to copy.
                            # Let's just check if the residue at that position is H.
                            # Since we don't have the map, we can just check if the position is in the chain.
                            # Actually, we can pre-compute the indices? No.
                            # Let's just iterate over the chain to find the index.
                            # This makes beam search O(N^2 * BeamWidth).
                            # For N=48, Beam=20, 48^2 * 20 * 48 steps? No, 48 steps * 20 beams * 48 neighbors * 48 chain length = ~900k ops per step. 48 steps * 900k = 43M ops. Should be fast enough in Python? Maybe tight.
                            
                            # Better: maintain a dictionary pos->idx in the state?
                            # That doubles memory and copy time.
                            # Let's try a different heuristic: just count potential contacts.
                            # Actually, the energy is just -contacts.
                            # We can just count how many H neighbors are already in the occupied set (excluding the immediate predecessor).
                            # But we need to know if those neighbors are H residues.
                            # Since the sequence is fixed, we know which positions are H.
                            # But we don't know which positions are in the chain yet.
                            # However, the chain is the prefix of the sequence.
                            # So the residues in the chain are sequence[0..step].
                            # The new residue is sequence[step].
                            # The neighbors q in new_occupied (excluding last) correspond to some indices j < step.
                            # We need to check if sequence[j] == 'H'.
                            # We can find j by looking up q in the chain.
                            
                            # Let's optimize by storing the chain as a list and using a set for occupied.
                            # To find index j, we can just check if sequence[j] is H.
                            # But we don't have the index j.
                            # We can store the chain as a list of (pos, idx)? No, idx is implicit.
                            # Let's just do the linear search for now. It's simple.
                            
                            found_h = False
                            for j, p in enumerate(new_chain[:-1]): # Exclude the new one? No, q is in occupied, so q is in new_chain.
                                if p == q:
                                    if sequence[j] == 'H':
                                        found_h = True
                                    break
                            if found_h:
                                new_score += 1
                                
                # Score is the number of contacts so far.
                # We want to maximize this.
                heapq.heappush(new_heap, (-new_score, counter, new_chain, new_occupied))
                counter += 1
                
                # Prune new_heap to beam_width
                if len(new_heap) > beam_width * 2: # Allow some slack before pruning
                    new_heap = heapq.nsmallest(beam_width, new_heap)
        
        heap = heapq.nsmallest(beam_width, new_heap)
        
    if not heap:
        # Fallback to random or linear
        return _random_fold(sequence, dimension, rng)
    
    # Get the best one
    best_score, _, best_chain, _ = heap[0]
    return best_chain

def _random_fold(sequence, dimension, rng):
    """
    Generate a random self-avoiding walk.
    Returns None if stuck.
    """
    n = len(sequence)
    if n == 0:
        return []
    chain = [(0,) * dimension]
    occupied = {chain[0]: 0}
    
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site not in occupied:
                options.append(site)
        if not options:
            return None
        site = rng.choice(options)
        occupied[site] = i
        chain.append(site)
    return chain

def _pivot_move(chain, pivot_idx, rng, dimension):
    """
    Perform a pivot move.
    Returns new chain or None if invalid.
    """
    n = len(chain)
    if n < 4 or pivot_idx < 1 or pivot_idx >= n - 1:
        return None
    
    p_prev = chain[pivot_idx - 1]
    p_pivot = chain[pivot_idx]
    
    # Determine bond axis
    v = [p_pivot[i] - p_prev[i] for i in range(dimension)]
    bond_axis = -1
    for i in range(dimension):
        if v[i] != 0:
            bond_axis = i
            break
    if bond_axis == -1:
        return None
        
    rot_axes = [i for i in range(dimension) if i != bond_axis]
    if not rot_axes:
        return None
        
    r_axis = rng.choice(rot_axes)
    angle = rng.choice([1, -1])
    
    head_set = set(chain[:pivot_idx])
    new_tail = []
    occupied_tail = set()
    
    for i in range(pivot_idx + 1, n):
        orig = chain[i]
        dx = orig[0] - p_pivot[0]
        dy = orig[1] - p_pivot[1]
        dz = orig[2] - p_pivot[2]
        
        if r_axis == 0: # Rotate around X
            if angle == 1:
                nx, ny, nz = dx, -dz, dy
            else:
                nx, ny, nz = dx, dz, -dy
        elif r_axis == 1: # Rotate around Y
            if angle == 1:
                nx, ny, nz = dz, dy, -dx
            else:
                nx, ny, nz = -dz, dy, dx
        else: # Rotate around Z
            if angle == 1:
                nx, ny, nz = -dy, dx, dz
            else:
                nx, ny, nz = dy, -dx, dz
                
        new_point = (p_pivot[0] + nx, p_pivot[1] + ny, p_pivot[2] + nz)
        
        if new_point in head_set:
            return None
        if new_point in occupied_tail:
            return None
            
        if new_tail:
            prev = new_tail[-1]
            if abs(new_point[0]-prev[0]) + abs(new_point[1]-prev[1]) + abs(new_point[2]-prev[2]) != 1:
                return None
                
        occupied_tail.add(new_point)
        new_tail.append(new_point)
        
    return chain[:pivot_idx + 1] + new_tail

def _kink_move(chain, kink_idx, rng, dimension):
    """
    Perform a kink move (move a single residue if possible).
    Returns new chain or None if invalid.
    """
    n = len(chain)
    if n < 5 or kink_idx < 2 or kink_idx > n - 3:
        return None
        
    p_prev = chain[kink_idx - 1]
    p_curr = chain[kink_idx]
    p_next = chain[kink_idx + 1]
    
    # Find common neighbours of p_prev and p_next that are not p_curr
    candidates = []
    for site in _neighbours(p_prev):
        if site == p_curr:
            continue
        # Check if adjacent to p_next
        is_adj_next = False
        for q in _neighbours(p_next):
            if q == site:
                is_adj_next = True
                break
        if is_adj_next:
            # Check collision
            valid = True
            for i, p in enumerate(chain):
                if i == kink_idx:
                    continue
                if p == site:
                    valid = False
                    break
            if valid:
                candidates.append(site)
                
    if not candidates:
        return None
        
    new_site = rng.choice(candidates)
    new_chain = chain[:kink_idx] + [new_site] + chain[kink_idx+1:]
    return new_chain

def _sa_fold(sequence, dimension, rng, deadline, initial_chain=None):
    """
    Simulated Annealing with Pivot and Kink moves.
    """
    n = len(sequence)
    if n == 0:
        return [], 0
    
    if initial_chain is None:
        chain = _random_fold(sequence, dimension, rng)
        if chain is None:
            chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    else:
        chain = list(initial_chain)
        
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # SA Parameters
    start_time = time.time()
    t0 = start_time
    duration = deadline - t0
    if duration <= 0:
        return best_chain, best_energy
        
    T_start = 5.0
    T_end = 0.01
    
    while time.time() < deadline - 0.1:
        elapsed = time.time() - t0
        progress = elapsed / duration
        if progress > 1.0:
            break
        T = T_start * (T_end / T_start) ** progress
        
        # Choose move type
        # 30% Kink, 70% Pivot
        if rng.random() < 0.3:
            kink_idx = rng.randint(2, n - 3)
            new_chain = _kink_move(chain, kink_idx, rng, dimension)
        else:
            pivot_idx = rng.randint(1, n - 2)
            new_chain = _pivot_move(chain, pivot_idx, rng, dimension)
            
        if new_chain is None:
            continue
            
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Uses a global time budget strategy.
    1. Generate an initial chain via Beam Search (deterministic/greedy).
    2. Optimize it using Simulated Annealing.
    3. If time permits, do a second restart with a random initial chain and SA.
    4. Keep the best result.
    
    Total time limit: 330s wall-clock for the whole program.
    Since fold is called once per sequence, and there are 26 sequences,
    we should reserve a global deadline.
    However, the contract says "240 s CPU budget for all sequences together" and "300s wall-clock".
    Let's assume the 240s is the hard limit for the sum of all calls.
    To be safe, let's track the time used by previous calls? 
    No, the subprocess is killed after 330s wall-clock.
    If we don't know the order, we can't easily share state.
    BUT, we can use a heuristic: 
    If we assume the calls are sequential, we can estimate the remaining time.
    However, a safer approach for a single call is to use a fixed budget that sums to < 240s.
    240 / 26 ≈ 9.2 seconds.
    Let's use 8.0 seconds per call to be safe.
    """
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0
    start_time = time.time()
    
    # Limit each call to 8.0 seconds to fit 26 calls in 208s (with margin).
    call_deadline = time.time() + 8.0
    
    n = len(sequence)
    
    # Phase 1: Beam Search Initial Fold
    # Beam width depends on length.
    # For short chains (N<50), beam width 50.
    # For long chains, beam width 20 to save time.
    beam_width = 50 if n < 50 else 20
    beam_deadline = time.time() + 2.0 # Spend up to 2s on beam search
    
    initial_chain = _beam_search_fold(sequence, dimension, beam_width, beam_deadline, rng)
    
    # Phase 2: SA Optimization on Beam Search Result
    # Remaining time: 8.0 - elapsed
    elapsed = time.time() - start_time
    remaining = call_deadline - time.time()
    
    if remaining > 1.0:
        # Use 70% of remaining time for SA on the beam search result
        sa_deadline_1 = time.time() + remaining * 0.7
        opt_chain, opt_energy = _sa_fold(sequence, dimension, rng, sa_deadline_1, initial_chain)
        
        if opt_chain is not None:
            if best_chain is None or opt_energy < best_energy:
                best_chain = opt_chain
                best_energy = opt_energy
        
        # Phase 3: SA Optimization on Random Result (if time permits)
        elapsed = time.time() - start_time
        remaining = call_deadline - time.time()
        
        if remaining > 1.0:
            # Use remaining time for a random restart
            sa_deadline_2 = time.time() + remaining * 0.9
            opt_chain, opt_energy = _sa_fold(sequence, dimension, rng, sa_deadline_2, None)
            
            if opt_chain is not None:
                if best_chain is None or opt_energy < best_energy:
                    best_chain = opt_chain
                    best_energy = opt_energy

    if best_chain is None:
        # Fallback: linear chain
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
