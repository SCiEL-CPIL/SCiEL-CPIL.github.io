# EVOLVE-BLOCK-START
import random
import time
import math
import sys

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """
    Compute the energy (negative number of H-H contacts) of a chain.
    """
    if not chain:
        return 0
    # Map position to index for O(1) lookup
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    for i in range(n):
        if sequence[i] == 'H':
            p = chain[i]
            for q in _neighbours(p):
                j = pos_to_idx.get(q)
                if j is not None and abs(i - j) > 1:
                    if sequence[j] == 'H':
                        contacts += 1
    # Energy is defined as -contacts in the prompt description 
    # "Energy = minus the number of H-H pairs..."
    # So lower energy (more negative) is better.
    return -contacts

def _random_fold(sequence, dimension, rng):
    """
    Generate a random self-avoiding walk.
    If it gets stuck, it returns None.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    start = (0, 0, 0)
    chain = [start]
    occupied = {start}
    
    for i in range(1, n):
        neighbors = list(_neighbours(chain[-1]))
        random.shuffle(neighbors)
        valid = False
        for nb in neighbors:
            if nb not in occupied:
                chain.append(nb)
                occupied.add(nb)
                valid = True
                break
        if not valid:
            return None
    return chain

def _pivot_move(chain, pivot_idx, dimension, rng):
    """
    Perform a pivot move at pivot_idx.
    Returns the new chain if valid, else None.
    """
    n = len(chain)
    if n < 3:
        return None
    
    # Pivot must be between 1 and n-2 to have a head and tail
    if pivot_idx < 1 or pivot_idx > n - 2:
        return None
        
    p_prev = chain[pivot_idx - 1]
    p_pivot = chain[pivot_idx]
    
    # Determine the bond vector from prev to pivot
    v = [p_pivot[i] - p_prev[i] for i in range(dimension)]
    bond_axis = -1
    for i in range(dimension):
        if v[i] != 0:
            bond_axis = i
            break
    if bond_axis == -1:
        return None
        
    # Possible rotation axes are the other 2 dimensions
    rot_axes = [i for i in range(dimension) if i != bond_axis]
    r_axis = rng.choice(rot_axes)
    angle = rng.choice([1, -1]) # 90 degree rotation
    
    # The head is chain[0:pivot_idx]
    # The tail is chain[pivot_idx+1:]
    # We rotate the tail around the pivot point.
    
    head_set = set(chain[:pivot_idx])
    # Note: p_pivot is part of the fixed part for the rotation axis, 
    # but the first atom of the tail (chain[pivot_idx+1]) is attached to p_pivot.
    # Actually, standard pivot: rotate tail (indices > pivot_idx) around the bond (pivot_idx-1, pivot_idx).
    # Wait, usually pivot is defined by the bond. Let's stick to rotating the tail part (indices >= pivot_idx+1) 
    # around the bond ending at pivot_idx.
    # The bond is between pivot_idx-1 and pivot_idx.
    # The atom at pivot_idx stays fixed. The atom at pivot_idx+1 moves.
    
    new_tail = []
    # The first atom of the tail is chain[pivot_idx+1]
    # Let's compute the new position for each atom in the tail relative to p_pivot.
    
    valid = True
    occupied_tail = set()
    
    # Pre-compute relative coordinates for the tail? 
    # For performance, we can just transform each point.
    
    for i in range(pivot_idx + 1, n):
        orig = chain[i]
        dx = orig[0] - p_pivot[0]
        dy = orig[1] - p_pivot[1]
        dz = orig[2] - p_pivot[2]
        
        # Apply rotation around r_axis
        if r_axis == 0: # Rotate around X
            if angle == 1: # +90 deg
                nx, ny, nz = dx, -dz, dy
            else: # -90 deg
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
        
        # Check self-avoidance
        if new_point in head_set:
            valid = False
            break
        if new_point in occupied_tail:
            valid = False
            break
            
        occupied_tail.add(new_point)
        new_tail.append(new_point)
        
    if not valid:
        return None
        
    return chain[:pivot_idx + 1] + new_tail

def _sa_fold(sequence, dimension, time_limit, rng):
    """
    Simulated Annealing with Pivot Moves.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    # 1. Generate initial random valid fold
    current_chain = None
    max_attempts = 100
    for _ in range(max_attempts):
        current_chain = _random_fold(sequence, dimension, rng)
        if current_chain:
            break
            
    if current_chain is None:
        # Fallback: straight line if random fails (should be rare for small N, but possible for large)
        # Actually, random SAWs often fail for N > 20-30. 
        # Let's use a simple greedy random walk that backtracks? No, just return a linear chain if possible?
        # A linear chain is always valid.
        current_chain = [(i, 0, 0) for i in range(n)]
        
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # SA Parameters
    # T0 needs to be high enough to accept bad moves initially.
    # Energy scale is roughly -N/2. Let's say T0 = 5.0
    T0 = 5.0
    Tend = 0.01
    # Cooling schedule: Geometric
    # We need to determine the number of steps or cooling rate.
    # Let's run for a fixed number of steps or until time limit.
    # Given 240s total for 26 chains, ~9s per chain.
    # SA is fast. We can do many steps.
    
    start_time = time.time()
    steps = 0
    
    # Estimate steps per second: ~100k-1M depending on N and optimization.
    # Let's just loop until time limit.
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 1.0: # Safety margin
            break
            
        # Calculate temperature
        # We can use a time-based cooling or step-based.
        # Step-based is safer if step time varies.
        # Let's assume we want to cool over ~50,000 steps.
        # If we run faster, we finish earlier.
        # If we run slower, we might not reach Tend, but that's okay.
        
        # Better: Use a dynamic cooling rate or just a large number of steps.
        # Let's define total_steps = 50000.
        # If we exceed time, we break.
        
        if steps == 0:
            # Initialize total steps based on N? 
            # Larger N needs more steps to explore.
            pass 
            
        # Let's use a fixed number of steps per chain, scaled by N?
        # Or just run until time limit.
        # To ensure cooling, we need to know total steps.
        # Let's estimate: 100,000 steps.
        
        # Actually, let's just use a simple geometric cooling with a fixed number of iterations.
        # If we break early due to time, T will be higher than Tend, which is fine.
        
        total_steps_target = 20000 + n * 100
        if steps >= total_steps_target:
             break
             
        T = T0 * (Tend / T0) ** (steps / float(total_steps_target))
        
        # Choose pivot
        pivot_idx = rng.randint(1, n - 2)
        new_chain = _pivot_move(current_chain, pivot_idx, dimension, rng)
        
        if new_chain is None:
            steps += 1
            continue
            
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        # Accept if better, or with probability exp(-delta_e / T)
        if delta_e < 0:
            current_chain = new_chain
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(new_chain)
        else:
            # delta_e >= 0 (worse or equal)
            if T > 0:
                prob = math.exp(-delta_e / T)
                if rng.random() < prob:
                    current_chain = new_chain
                    current_energy = new_energy
            # else: T is 0, only accept better
            
        steps += 1
        
    return best_chain

def fold(sequence, dimension):
    """
    Multi-Start Simulated Annealing.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    # Total budget 240s. 26 sequences.
    # We have ~9s per sequence.
    # Let's allocate time for multiple SA runs.
    # SA is stochastic, so multiple restarts help.
    
    total_time_limit = 8.5 # Leave 0.5s for overhead/conversion
    start_time = time.time()
    
    # Determine number of restarts
    # Let's try to run SA for at least 1 second each, or more if possible.
    # For small N, SA is very fast. For large N, it's slower.
    # Let's do 3 restarts.
    
    best_chain_overall = None
    best_energy_overall = float('inf')
    
    # We can adjust time per run based on N
    # Smaller N: more restarts. Larger N: fewer, longer runs.
    # Let's stick to 3 runs of ~2.5s each.
    
    num_restarts = 3
    time_per_run = total_time_limit / num_restarts
    
    for r in range(num_restarts):
        remaining = total_time_limit - (time.time() - start_time)
        if remaining < 0.5:
            break
            
        # Use a different seed for each restart
        rng = random.Random(12345 + len(sequence) * 100 + r)
        
        # Run SA
        # If this is the last run, use all remaining time
        t_limit = time_per_run if r < num_restarts - 1 else remaining
        
        chain = _sa_fold(sequence, dimension, t_limit, rng)
        
        if chain:
            energy = _energy(sequence, chain)
            if energy < best_energy_overall:
                best_energy_overall = energy
                best_chain_overall = chain
                
    if best_chain_overall is None:
        # Fallback
        best_chain_overall = [(i, 0, 0) for i in range(n)]
        
    # Convert to list of lists
    return [list(p) for p in best_chain_overall]

# EVOLVE-BLOCK-END
