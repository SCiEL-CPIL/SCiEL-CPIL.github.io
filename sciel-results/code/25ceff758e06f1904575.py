# EVOLVE-BLOCK-START
import random
import time
import sys
import math

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
    
    # The residue at kink_idx must move to a new position
    # It must be adjacent to p_prev and p_next
    # And not collide with the rest of the chain
    
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
            # The rest of the chain is chain[:kink_idx-1] + chain[kink_idx+1:]
            # We need to ensure site is not in that set
            # Optimization: just check against the whole chain except kink_idx
            # But we must be careful not to include p_prev and p_next in the collision check
            # Actually, p_prev and p_next are in the chain. 
            # If site == p_prev or site == p_next, it's invalid because it would overlap.
            # But we already excluded p_curr. 
            # If site is p_prev, it's adjacent to p_prev (distance 0? No, neighbour). 
            # Wait, _neighbours(p_prev) yields sites at distance 1. So site != p_prev.
            # Similarly site != p_next.
            
            # Check against all other residues
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

def _sa_fold(sequence, dimension, rng, deadline):
    """
    Simulated Annealing with Pivot and Kink moves.
    """
    n = len(sequence)
    if n == 0:
        return [], 0
    
    # Initial random fold
    chain = None
    while chain is None:
        chain = _random_fold(sequence, dimension, rng)
        if chain is None:
            # If random fold fails (stuck), try a different random seed or break
            # For robustness, if stuck, we might need to retry with different random start
            # But _random_fold uses rng. If it's stuck, it returns None.
            # We should probably just break and return a linear fold if we can't get a random one.
            # However, for small N, random fold rarely fails. For large N, it might.
            # Let's try a few times.
            if rng.randint(0, 100) > 90: # Simple heuristic to avoid infinite loop
                 # Fallback to linear
                 chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
                 break
    
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # SA Parameters
    # Time-based cooling
    start_time = time.time()
    t0 = time.time()
    duration = deadline - t0
    if duration <= 0:
        return best_chain, best_energy
        
    T_start = 5.0
    T_end = 0.01
    alpha = (T_end / T_start) ** (1.0 / 1000.0) # Rough guess, we'll adjust based on time
    
    # We don't know exactly how many steps we can take.
    # Let's estimate steps per second.
    # But simpler: use time to determine T.
    # T = T_start * (T_end/T_start) ** (elapsed/duration)
    
    while time.time() < deadline - 0.1: # Safety margin
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
    1. Generate multiple initial chains via randomized growth.
    2. Optimize each one for a fixed amount of time.
    3. Keep the best result.
    
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
    
    n_restarts = 5
    # For long chains, maybe fewer restarts but longer optimization?
    # Or just fixed time per restart?
    # Let's give each restart a slice of the time budget.
    
    for i in range(n_restarts):
        if time.time() > call_deadline - 0.5: # Leave 0.5s for last one to finish
            break
            
        # Optimization budget for this restart
        # Simple strategy: each restart gets (call_deadline - start_time) / n_restarts
        # But if we are late, give more to the current one?
        # Let's just give a fixed time slice, say 1.5s per restart? 5 * 1.5 = 7.5s.
        # This leaves 0.5s margin.
        
        opt_deadline = time.time() + 1.5
        
        optimized_chain, opt_energy = _sa_fold(sequence, dimension, rng, opt_deadline)
        
        if optimized_chain is not None:
            if best_chain is None or opt_energy < best_energy:
                best_chain = optimized_chain
                best_energy = opt_energy

    if best_chain is None:
        # Fallback: linear chain
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
