"""Gravitational Search Algorithm for flow-shop permutation problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, p):
    """Compute makespan for a given permutation."""
    n = len(perm)
    m = p.shape[1]
    # completion time matrix: C[i][j] = completion time of job i on machine j
    # We process jobs in order 'perm', machines 0..m-1
    # For each job in sequence, its start on machine j is max(completion on machine j-1, completion of previous job on machine j)
    prev_machine_completion = np.zeros(m)
    makespan = 0.0
    for job_idx in perm:
        prev_job_completion = 0.0
        for j in range(m):
            # start time on machine j for this job
            start = max(prev_job_completion, prev_machine_completion[j])
            prev_job_completion = start + p[job_idx, j]
            prev_machine_completion[j] = prev_job_completion
        makespan = prev_job_completion
    return makespan


def _neh_order(p):
    """NEH heuristic: order jobs by decreasing total processing time,
    inserting each job in the best position."""
    n = p.shape[0]
    totals = p.sum(axis=1)
    order = np.argsort(-totals)
    
    # Start with empty schedule
    schedule = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(schedule) + 1):
            candidate = schedule[:pos] + [job] + schedule[pos:]
            ms = _makespan(candidate, p)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        schedule.insert(best_pos, job)
    return schedule


def _two_opt(perm, p, max_iters=100):
    """Simple 2-opt improvement: swap adjacent jobs if it improves makespan."""
    perm = list(perm)
    n = len(perm)
    current_ms = _makespan(perm, p)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(n - 1):
            # Try swapping perm[i] and perm[i+1]
            new_perm = perm[:]
            new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
            new_ms = _makespan(new_perm, p)
            if new_ms < current_ms:
                perm = new_perm
                current_ms = new_ms
                improved = True
                # Restart scan from beginning after a swap
                break
    return perm, current_ms


def order_jobs(processing_times):
    p = np.asarray(processing_times, dtype=np.float64)
    n, m = p.shape
    
    if n <= 2:
        return list(range(n))
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initialize with NEH heuristic
    best_perm = _neh_order(p)
    best_ms = _makespan(best_perm, p)
    
    # Apply 2-opt improvement
    best_perm, best_ms = _two_opt(best_perm, p, max_iters=200)
    
    # Gravitational Search Algorithm
    # Population of permutations, each with velocity (position changes)
    pop_size = min(30, max(10, n * 2))
    
    # Initialize population: NEH + random perturbations
    population = [best_perm[:]]
    for i in range(pop_size - 1):
        if time.time() > deadline:
            break
        # Create perturbation of best
        perm = best_perm[:]
        # Random swaps
        num_swaps = max(1, n // 5)
        for _ in range(num_swaps):
            i1 = np.random.randint(0, n)
            i2 = np.random.randint(0, n)
            perm[i1], perm[i2] = perm[i2], perm[i1]
        population.append(perm)
    
    # Compute makespans for population
    pop_ms = [_makespan(perm, p) for perm in population]
    
    # Track best
    global_best_idx = np.argmin(pop_ms)
    global_best_perm = population[global_best_idx][:]
    global_best_ms = pop_ms[global_best_idx]
    
    # GSA parameters
    G = 1.0  # gravitational constant
    alpha = 0.5  # exponential rate parameter for G
    max_iter = 50
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Update G
        G_iter = G * np.exp(-alpha * iteration / max_iter)
        
        # Compute masses (inversely proportional to makespan, normalized)
        # Better solutions have larger mass
        inv_ms = 1.0 / (np.array(pop_ms) + 1e-10)
        total_inv = inv_ms.sum()
        masses = inv_ms / total_inv  # normalized masses
        
        # For each particle, compute gravitational force from all other particles
        # In permutation space, we interpret "position" as the permutation itself
        # Force direction: move towards better particles (higher mass)
        
        for i in range(pop_size):
            if time.time() > deadline:
                break
            
            # Compute net force direction: weighted average of positions of other particles
            # weighted by their mass and "distance" (inversion distance approximation)
            force_perm = list(range(n))
            force_weights = np.zeros(n)
            
            for j in range(pop_size):
                if i == j:
                    continue
                
                # Compute a simple distance metric (number of inversions difference)
                # Use Kendall tau-like distance approximation
                dist = 0
                for k in range(n):
                    # Count how many elements differ in position
                    if population[i][k] != population[j][k]:
                        dist += 1
                dist = dist / n  # normalize to [0, 1]
                
                # Force is proportional to mass_j * (1 - dist) - attraction to better, repulsion from worse
                # Actually in GSA, force is attractive: F = G * m_i * m_j / (r^2 + eps) * (x_j - x_i)
                # In permutation space, we approximate: pull towards j if j is better
                weight = G_iter * masses[j] / (dist + 0.1)
                
                if pop_ms[j] < pop_ms[i]:
                    # Attract towards j
                    force_weights += weight * np.array(population[j])
                else:
                    # Slight repulsion from worse
                    force_weights -= 0.1 * weight * np.array(population[j])
            
            # Apply force to create a new candidate permutation
            # Use the force to guide swaps: positions with high force weight get moved
            new_perm = population[i][:]
            
            # Identify top positions to potentially swap
            if force_weights.sum() > 0:
                # Get indices sorted by force weight
                top_idx = np.argsort(-force_weights)[:n // 2]
                for idx in top_idx:
                    if time.time() > deadline:
                        break
                    # Try to move job at position idx towards where it is in the weighted target
                    job = new_perm[idx]
                    target_pos = np.where(np.array(population[global_best_idx]) == job)[0]
                    if len(target_pos) > 0:
                        target_pos = target_pos[0]
                        if target_pos != idx:
                            # Swap or insert
                            new_perm = new_perm[:idx] + new_perm[idx+1:target_pos+1] + [job] + new_perm[target_pos+1:idx+1] if target_pos > idx else new_perm[:target_pos] + new_perm[target_pos+1:idx+1] + [job] + new_perm[idx+1:target_pos+1]
            
            # Evaluate new candidate
            new_ms = _makespan(new_perm, p)
            
            # Accept if better or with some probability (simulated annealing style)
            if new_ms < pop_ms[i]:
                population[i] = new_perm
                pop_ms[i] = new_ms
            else:
                # With small probability, accept worse to escape local optima
                if np.random.random() < 0.1 * np.exp(-iteration / max_iter):
                    population[i] = new_perm
                    pop_ms[i] = new_ms
            
            # Update global best
            if pop_ms[i] < global_best_ms:
                global_best_ms = pop_ms[i]
                global_best_perm = population[i][:]
        
        # Elite preservation: keep best in population
        if pop_ms[global_best_idx] > global_best_ms:
            # Find current best in population
            current_best_idx = np.argmin(pop_ms)
            if pop_ms[current_best_idx] < global_best_ms:
                global_best_ms = pop_ms[current_best_idx]
                global_best_perm = population[current_best_idx][:]
        
        # Replace worst with perturbation of global best
        worst_idx = np.argmax(pop_ms)
        new_worst = global_best_perm[:]
        for _ in range(max(1, n // 10)):
            i1 = np.random.randint(0, n)
            i2 = np.random.randint(0, n)
            new_worst[i1], new_worst[i2] = new_worst[i2], new_worst[i1]
        population[worst_idx] = new_worst
        pop_ms[worst_idx] = _makespan(new_worst, p)
    
    # Final 2-opt refinement on best
    final_perm, final_ms = _two_opt(global_best_perm, p, max_iters=300)
    
    if time.time() > deadline:
        return list(global_best_perm)
    
    return list(final_perm)
# EVOLVE-BLOCK-END
