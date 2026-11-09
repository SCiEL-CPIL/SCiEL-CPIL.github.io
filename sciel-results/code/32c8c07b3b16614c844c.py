"""
Multi-Start Discrete Particle Swarm Optimization for Flow-Shop Scheduling.
Uses a vectorized makespan evaluator for speed and Or-opt moves for local refinement.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _fast_makespan(order, P):
    """
    Compute makespan using vectorized operations.
    P: (n_jobs, n_machines)
    order: list or array of job indices
    """
    if len(order) == 0:
        return 0.0
    
    # Get processing times for the specific order
    # Shape: (n_jobs_in_order, n_machines)
    P_order = P[order]
    
    # Cumulative sum along machines axis
    # cumsum[i, m] = sum(P_order[i, 0:m+1])
    cumsum = np.cumsum(P_order, axis=1)
    
    # We need completion time C[i, m] = max(C[i, m-1], C[i-1, m]) + P[i, m]
    # This is hard to do in one vectorized op without loops over jobs if M is large,
    # but if M is small (typical flow shop), a loop over jobs with vectorized machine ops is fast.
    # Alternatively, for small M, we can just loop. Let's try a hybrid:
    # If M is very small, pure python loop might be competitive, but numpy overhead is high.
    # Let's use a loop over jobs but vectorize the machine update?
    # Actually, the standard fast trick is:
    # C[i, m] = max(C[i-1, m], C[i, m-1]) + P[i, m]
    
    # Let's stick to a tight Python loop for the recurrence, which is usually fast enough for N<100, M<20.
    # But we can optimize the inner loop.
    
    n_jobs = len(order)
    n_machines = P.shape[1]
    
    # Pre-allocate completion array
    # C[i, m] is completion time of job i on machine m
    # We only need the previous row
    prev_row = np.zeros(n_machines)
    
    for i in range(n_jobs):
        p_i = P_order[i]
        # C[i, 0] = C[i-1, 0] + P[i, 0]
        # C[i, m] = max(C[i-1, m], C[i, m-1]) + P[i, m]
        
        # Vectorized update for m=1..M-1
        # current_row[0] = prev_row[0] + p_i[0]
        # current_row[m] = max(prev_row[m], current_row[m-1]) + p_i[m]
        
        # This dependency chain (current_row[m] depends on current_row[m-1]) prevents full vectorization per job.
        # However, for small M, a Python loop over M is fine.
        
        curr_row = np.empty(n_machines)
        curr_row[0] = prev_row[0] + p_i[0]
        for m in range(1, n_machines):
            curr_row[m] = max(prev_row[m], curr_row[m-1]) + p_i[m]
        
        prev_row = curr_row
        
    return float(prev_row[-1])


def _or_opt_move(order, P, best_order, best_ms):
    """
    Try Or-opt moves (moving a block of 1, 2, or 3 jobs) to improve the order.
    Returns improved order and makespan if found.
    """
    n = len(order)
    current_order = list(order)
    current_ms = best_ms
    
    # Try block sizes 1, 2, 3
    for block_size in [1, 2, 3]:
        for i in range(n - block_size):
            # Extract block
            block = current_order[i:i+block_size]
            # Remove block
            remaining = current_order[:i] + current_order[i+block_size:]
            
            # Try inserting block at every position in remaining
            # We can optimize by skipping positions that are equivalent or clearly worse, 
            # but for simplicity and speed, we check all valid positions.
            
            # Valid insertion positions are 0 to len(remaining)
            # Note: Inserting at i or i+block_size returns to original (mostly)
            
            for j in range(len(remaining) + 1):
                # Skip if it results in the same permutation (rough check)
                if j == i or j == i + block_size:
                    continue
                    
                new_order = remaining[:j] + block + remaining[j:]
                
                # Quick check: if the block is moved far, compute makespan
                ms = _fast_makespan(new_order, P)
                
                if ms < current_ms - 1e-9:
                    current_order = new_order
                    current_ms = ms
    
    return current_order, current_ms


def _discrete_velocity_update(particle, global_best, local_best, w, c1, c2, rng):
    """
    Update a particle's position using discrete PSO rules.
    Velocity is a probability of moving a job from its current position to the position in the best.
    """
    n = len(particle)
    new_particle = list(particle)
    
    # For each job in the particle, consider moving it to where it is in global_best or local_best
    # Standard discrete PSO: v = w*v + c1*r1*(pb - x) + c2*r2*(gb - x)
    # In permutation space, 'difference' is often handled by inserting the job from the best at its best position.
    
    # Simplified Discrete PSO:
    # 1. With probability w, do nothing (inertia).
    # 2. With probability c1, swap/insert job from local_best.
    # 3. With probability c2, swap/insert job from global_best.
    
    # A common effective operator:
    # Pick a random job index k.
    # If rng() < c1:
    #    Take the job at position k in local_best and insert it at position k in new_particle (shifting others).
    # If rng() < c2:
    #    Take the job at position k in global_best and insert it at position k in new_particle.
    
    # Let's use the "insertion" based update which is robust.
    
    for k in range(n):
        # Decide if we update position k based on cognitive and social factors
        r1 = rng.random()
        r2 = rng.random()
        
        # Probability of updating this position from local best
        if r1 < c1:
            job_to_insert = local_best[k]
            # Remove job_to_insert from current new_particle
            try:
                idx = new_particle.index(job_to_insert)
                new_particle.pop(idx)
            except ValueError:
                # Should not happen if permutations are valid
                pass
            # Insert at position k
            new_particle.insert(k, job_to_insert)
            
        # Probability of updating this position from global best
        if r2 < c2:
            job_to_insert = global_best[k]
            try:
                idx = new_particle.index(job_to_insert)
                new_particle.pop(idx)
            except ValueError:
                pass
            new_particle.insert(k, job_to_insert)
            
    return new_particle


def order_jobs(processing_times):
    P = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = P.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    # 1. Initial Solution: NEH Heuristic (High quality start)
    # Sort by descending total processing time
    totals = P.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    neh_order = []
    for job in sorted_jobs:
        best_pos = 0
        best_ms = np.inf
        for pos in range(len(neh_order) + 1):
            trial_order = neh_order[:pos] + [job] + neh_order[pos:]
            ms = _fast_makespan(trial_order, P)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        neh_order.insert(best_pos, job)
        
    global_best_order = list(neh_order)
    global_best_ms = _fast_makespan(global_best_order, P)
    
    # 2. Initialize Swarm
    swarm_size = 20  # Adjust based on n_jobs
    if n_jobs > 50:
        swarm_size = 10
    
    particles = []
    local_bests = []
    local_best_ms = []
    
    # Seed one particle with NEH, others random
    for i in range(swarm_size):
        if i == 0:
            p = list(neh_order)
        else:
            p = list(np.arange(n_jobs))
            rng.shuffle(p)
        
        ms = _fast_makespan(p, P)
        particles.append(p)
        local_bests.append(list(p))
        local_best_ms.append(ms)
        
        if ms < global_best_ms:
            global_best_ms = ms
            global_best_order = list(p)
    
    # 3. PSO Loop
    w = 0.7  # Inertia (not directly used in simple insertion update, but keeps diversity conceptually)
    c1 = 0.4  # Cognitive
    c2 = 0.4  # Social
    
    iteration = 0
    no_improve_count = 0
    
    while time.time() < deadline:
        iteration += 1
        
        # Update particles
        for i in range(swarm_size):
            # Discrete velocity update
            new_p = _discrete_velocity_update(
                particles[i], 
                global_best_order, 
                local_bests[i], 
                w, c1, c2, rng
            )
            
            ms = _fast_makespan(new_p, P)
            
            # Update local best
            if ms < local_best_ms[i]:
                local_best_ms[i] = ms
                local_bests[i] = list(new_p)
                
                # Update global best
                if ms < global_best_ms:
                    global_best_ms = ms
                    global_best_order = list(new_p)
                    no_improve_count = 0
            else:
                no_improve_count += 1
            
            particles[i] = new_p
        
        # Periodic Local Search (Or-opt) on Global Best
        # Do this every 10 iterations or if stuck
        if iteration % 10 == 0 or no_improve_count > 50:
            improved_order, improved_ms = _or_opt_move(global_best_order, P, global_best_order, global_best_ms)
            if improved_ms < global_best_ms - 1e-9:
                global_best_ms = improved_ms
                global_best_order = improved_order
                no_improve_count = 0
                # Update the particle that matches this best? Or just let PSO continue.
                # It's better to inject this best back into the swarm to spread it.
                # Let's set the first particle to this best.
                particles[0] = list(global_best_order)
                local_bests[0] = list(global_best_order)
                local_best_ms[0] = global_best_ms
                
        # Check time
        if time.time() > deadline - 5.0:
            break
            
    return global_best_order
# EVOLVE-BLOCK-END
