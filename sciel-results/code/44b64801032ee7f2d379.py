"""HP lattice protein folding using pivot-based Simulated Annealing."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global state for time management
_START_TIME = time.time()
_TIME_LIMIT = 105.0 # Safety margin for 120s limit
_CALL_COUNT = 0
_TOTAL_CALLS = 10

def _neighbours(point):
    """Yield the 4 neighbors of a point on the square lattice."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _compute_energy(sequence, chain):
    """Compute the energy of a conformation.
    Energy = - (number of H-H contacts).
    Contacts are pairs (i, j) with i < j, |i-j| > 1, sequence[i]=='H', sequence[j]=='H', 
    and Manhattan distance between chain[i] and chain[j] is 1.
    """
    n = len(sequence)
    if n == 0:
        return 0
    # Map position to index for O(1) lookup
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        xi, yi = chain[i]
        # Check 4 neighbors
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbor = (xi + dx, yi + dy)
            j = pos_to_idx.get(neighbor)
            if j is not None and j > i + 1:
                if sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _grow_random(sequence, dimension, rng):
    """Generate a random self-avoiding walk (SAW)."""
    n = len(sequence)
    if n == 0:
        return []
    chain = [(0, 0)]
    occupied = {(0, 0)}
    for _ in range(1, n):
        last = chain[-1]
        options = [p for p in _neighbours(last) if p not in occupied]
        if not options:
            # Dead end, restart
            return _grow_random(sequence, dimension, rng)
        next_pos = rng.choice(options)
        chain.append(next_pos)
        occupied.add(next_pos)
    return chain

def _grow_greedy(sequence, dimension, rng):
    """Generate a SAW using a greedy heuristic to maximize H-H contacts."""
    n = len(sequence)
    if n == 0:
        return []
    
    best_chain = None
    best_score = -1000
    
    # Try a few greedy starts
    for _ in range(10):
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        current_score = 0
        
        for i in range(1, n):
            last = chain[-1]
            options = []
            for p in _neighbours(last):
                if p in occupied:
                    continue
                # Score this move
                # 1. Immediate H-H contact gain
                gain = 0
                if sequence[i] == 'H':
                    for q in _neighbours(p):
                        idx = occupied.get(q)
                        if idx is not None and idx < i - 1 and sequence[idx] == 'H':
                            gain += 1
                
                # 2. Free space heuristic (prefer moves with more open neighbors to avoid dead ends)
                free = sum(1 for q in _neighbours(p) if q not in occupied)
                
                # Combine: prioritize contacts, then free space, then random
                score = gain * 10 + free + rng.random()
                options.append((score, p))
            
            if not options:
                break
            
            # Pick the best option
            best_score_move, next_pos = max(options)
            chain.append(next_pos)
            occupied[next_pos] = i
            current_score += best_score_move
            
        if len(chain) == n:
            # Calculate actual energy to compare
            energy = _compute_energy(sequence, chain)
            # We want to maximize energy (less negative is better? No, energy is negative, so higher is better)
            # Wait, energy = -contacts. So -5 is better than -2.
            # So we want to maximize energy.
            if energy > best_score:
                best_score = energy
                best_chain = chain
        
        # Avoid infinite loop in greedy if it keeps failing
        if best_chain is not None and rng.random() < 0.1:
            break

    if best_chain is not None:
        return best_chain
    return _grow_random(sequence, dimension, rng)

def _pivot_move(chain, rng):
    """Perform a pivot move: rotate a sub-chain segment.
    Returns (new_chain, success).
    """
    n = len(chain)
    if n < 4:
        return None, False
    
    # Pick a pivot index i (1 to n-2)
    i = rng.randint(1, n - 2)
    # Pick a rotation angle (90, 180, 270 degrees)
    # Represented by direction vectors
    # Original vector from chain[i] to chain[i+1]
    v = (chain[i+1][0] - chain[i][0], chain[i+1][1] - chain[i][1])
    
    # Possible rotations of v
    if v == (1, 0): # Right
        rotations = [(0, 1), (-1, 0), (0, -1)] # Up, Left, Down
    elif v == (-1, 0): # Left
        rotations = [(0, -1), (1, 0), (0, 1)] # Down, Right, Up
    elif v == (0, 1): # Up
        rotations = [(-1, 0), (0, -1), (1, 0)] # Left, Down, Right
    elif v == (0, -1): # Down
        rotations = [(1, 0), (0, 1), (-1, 0)] # Right, Up, Left
    else:
        return None, False
        
    # Pick a rotation
    dx, dy = rng.choice(rotations)
    
    # Determine the new direction for the segment
    # The segment is chain[i+1:]
    # We need to map the relative coordinates of the segment based on the new direction.
    
    # Let's define the transformation.
    # If v was (1,0) and we rotate to (0,1):
    # (dx, dy) -> (-dy, dx) ? No.
    # Let's use complex numbers or explicit mapping.
    
    # Original vector v. New vector w = (dx, dy).
    # The rest of the chain is relative to chain[i].
    # Let rel_pos[k] = chain[i+k] - chain[i] for k >= 1.
    # We want to transform rel_pos such that rel_pos[1] becomes w.
    
    # If v=(1,0), w=(0,1):
    # (1,0) -> (0,1)
    # (0,1) -> (-1,0)
    # (0,-1) -> (1,0)
    # (-1,0) -> (0,-1)
    # This is a 90 deg CCW rotation.
    
    # If v=(1,0), w=(-1,0):
    # (1,0) -> (-1,0)
    # (0,1) -> (0,-1)
    # This is a 180 deg rotation.
    
    # If v=(1,0), w=(0,-1):
    # (1,0) -> (0,-1)
    # (0,1) -> (1,0)
    # This is a 90 deg CW rotation.
    
    # General mapping:
    # We can determine the transformation matrix by looking at v and w.
    # Let's just try to construct the new chain and check validity.
    
    new_chain = chain[:i+1]
    occupied = set(chain[:i+1])
    
    # We need to apply the rotation to the entire tail.
    # Let's find the transformation function f such that f(v) = w.
    # Since these are unit vectors on a square lattice, the transformation is a rotation.
    
    # Determine rotation type
    if v == (1, 0):
        if w == (0, 1): # CCW 90
            def transform(p):
                x, y = p
                return (-y, x)
        elif w == (-1, 0): # 180
            def transform(p):
                x, y = p
                return (-x, -y)
        else: # (0, -1) CW 90
            def transform(p):
                x, y = p
                return (y, -x)
    elif v == (-1, 0):
        if w == (0, -1): # CCW 90
            def transform(p):
                x, y = p
                return (-y, x)
        elif w == (1, 0): # 180
            def transform(p):
                x, y = p
                return (-x, -y)
        else: # (0, 1) CW 90
            def transform(p):
                x, y = p
                return (y, -x)
    elif v == (0, 1):
        if w == (-1, 0): # CCW 90
            def transform(p):
                x, y = p
                return (-y, x)
        elif w == (0, -1): # 180
            def transform(p):
                x, y = p
                return (-x, -y)
        else: # (1, 0) CW 90
            def transform(p):
                x, y = p
                return (y, -x)
    else: # v == (0, -1)
        if w == (1, 0): # CCW 90
            def transform(p):
                x, y = p
                return (-y, x)
        elif w == (0, 1): # 180
            def transform(p):
                x, y = p
                return (-x, -y)
        else: # (-1, 0) CW 90
            def transform(p):
                x, y = p
                return (y, -x)
                
    # Apply transformation to the tail
    tail = chain[i+1:]
    # Translate tail so chain[i] is origin
    xi, yi = chain[i]
    rel_tail = [(x - xi, y - yi) for x, y in tail]
    
    new_rel_tail = [transform(p) for p in rel_tail]
    
    # Translate back
    new_tail = [(x + xi, y + yi) for x, y in new_rel_tail]
    
    # Check validity
    valid = True
    for p in new_tail:
        if p in occupied:
            valid = False
            break
        # Also check if the new tail intersects itself
        # We add to a set to check self-intersection of the new tail
        # But we also need to ensure the first step is valid (adjacent to chain[i])
        # This is guaranteed by the rotation of v.
        
    if not valid:
        return None, False
    
    # Check self-intersection of the new tail itself
    seen = set(occupied)
    for p in new_tail:
        if p in seen:
            valid = False
            break
        seen.add(p)
        
    if not valid:
        return None, False
        
    # Construct final chain
    final_chain = new_chain + new_tail
    return final_chain, True

def _sa_search(sequence, initial_chain, rng, time_limit):
    """Simulated Annealing with pivot moves."""
    n = len(sequence)
    if n < 4:
        return initial_chain
    
    current_chain = list(initial_chain)
    current_energy = _compute_energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    temp = 1.0
    cooling = 0.9999
    
    # We need to track occupied positions for fast collision detection?
    # Pivot move generates a new list. Checking collision is O(N).
    # With N=100, O(N) is fine.
    
    while time.time() - start_time < time_limit:
        new_chain, success = _pivot_move(current_chain, rng)
        if not success:
            # Try a few more times before giving up on this step?
            # Or just continue to next iteration.
            # To avoid stalling, we can try 5 times.
            success = False
            for _ in range(5):
                new_chain, success = _pivot_move(current_chain, rng)
                if success:
                    break
            if not success:
                continue
            
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Acceptance criterion
        if delta > 0 or rng.random() < math.exp(delta / temp):
            # Note: delta is (new - old). If new is better (more negative), delta is negative.
            # Wait. Energy = -contacts.
            # Better energy = more negative.
            # If new_energy < current_energy, delta is negative.
            # We want to accept if delta < 0 (improvement) OR with probability exp(delta/temp).
            # If delta < 0, exp(delta/temp) < 1. We should accept with probability 1 if delta < 0.
            # So: if delta < 0: accept. else: accept with prob exp(delta/temp).
            
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                
        temp *= cooling
        if temp < 0.01:
            temp = 0.01
            
    return best_chain

def fold(sequence, dimension):
    """Best conformation using pivot-based SA."""
    global _CALL_COUNT
    _CALL_COUNT += 1
    
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    if n == 2:
        return [[0, 0], [1, 0]]
    if n == 3:
        return [[0, 0], [1, 0], [1, 1]]
        
    # Calculate time budget for this call
    elapsed = time.time() - _START_TIME
    time_left = _TIME_LIMIT - elapsed
    if time_left <= 0:
        # Fallback
        return [[i, 0] for i in range(n)]
        
    # Allocate time
    # We don't know exactly how many calls are left, but we assume 10 total.
    # If we are on call k, we have roughly (10 - k + 1) calls left including this one?
    # Actually, if we've done k-1 calls, we have 10 - (k-1) calls left.
    # So time_per_call = time_left / (10 - (k-1))
    calls_done = _CALL_COUNT - 1
    remaining_calls = max(1, 10 - calls_done)
    time_per_call = time_left / remaining_calls
    
    # Cap time per call to avoid huge spikes if we finished early
    # But if we have lots of time left, use it.
    
    rng = random.Random(12345 + len(sequence))
    
    # Multi-start SA
    # Use the available time for multiple restarts or one long run?
    # Pivot moves are powerful, so one long run might be good.
    # But multi-start helps escape global minima.
    # Let's split time: 50% for initial greedy, 50% for SA?
    # No, greedy is fast.
    # Let's do:
    # 1. Generate a few good initial conformations (Greedy + Random)
    # 2. Run SA on the best one for the remaining time.
    
    # Generate initial candidates
    candidates = []
    
    # 1. Greedy
    g_chain = _grow_greedy(sequence, dimension, rng)
    candidates.append(g_chain)
    
    # 2. Random SAWs
    for _ in range(3):
        r_chain = _grow_random(sequence, dimension, rng)
        if r_chain:
            candidates.append(r_chain)
            
    # Pick the best candidate
    best_init = None
    best_init_energy = 1000
    for c in candidates:
        if len(c) == n:
            e = _compute_energy(sequence, c)
            if e < best_init_energy:
                best_init_energy = e
                best_init = c
                
    if best_init is None:
        # Fallback
        best_init = [[i, 0] for i in range(n)]
        
    # Run SA
    # Use 80% of time_per_call for SA, keep 20% buffer for overhead
    sa_time = time_per_call * 0.8
    if sa_time < 0.1:
        sa_time = 0.1
        
    best_chain = _sa_search(sequence, best_init, rng, sa_time)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
