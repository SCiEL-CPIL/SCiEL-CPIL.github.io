"""Flow-shop permutation via multi-start heuristics + fast local search + discrete PSO."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan_fast(pt_list, perm):
    """Fast makespan calculation using pure Python lists."""
    n_machines = len(pt_list[0]) if pt_list else 0
    n = len(perm)
    if n == 0:
        return 0.0
    if n_machines == 0:
        return 0.0

    prev = [0.0] * n_machines

    for job in perm:
        row = pt_list[job]
        curr = [0.0] * n_machines
        curr[0] = prev[0] + row[0]
        for m in range(1, n_machines):
            val = prev[m] if prev[m] > curr[m-1] else curr[m-1]
            curr[m] = val + row[m]
        prev = curr
    return prev[-1]


def _makespan_vectorized(pt, perm):
    """Vectorized makespan using numpy."""
    n = len(perm)
    if n == 0:
        return 0.0
    n_machines = pt.shape[1]
    if n_machines == 0:
        return 0.0
    prev = np.zeros(n_machines)
    for job in perm:
        row = pt[job]
        curr = np.empty(n_machines)
        curr[0] = prev[0] + row[0]
        for m in range(1, n_machines):
            curr[m] = np.maximum(prev[m], curr[m-1]) + row[m]
        prev = curr
    return prev[-1]


def _neh_initial(pt):
    """NEH heuristic."""
    n_jobs, n_machines = pt.shape
    totals = pt.sum(axis=1)
    order = list(np.argsort(-totals))
    perm = [order[0]]
    pt_list = pt.tolist()
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            test = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_fast(pt_list, test)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, job)
    return perm


def _palmer_initial(pt):
    """Palmer's dispatching rule."""
    n_jobs, n_machines = pt.shape
    n = n_machines
    scores = np.zeros(n_jobs)
    for j in range(n_jobs):
        s = 0.0
        for m in range(n):
            if m < n // 2:
                s += (n - 1 - m) * pt[j, m]
            else:
                s -= m * pt[j, m]
        scores[j] = s
    return list(np.argsort(-scores))


def _lpt_initial(pt):
    totals = pt.sum(axis=1)
    return list(np.argsort(-totals))


def _spt_initial(pt):
    totals = pt.sum(axis=1)
    return list(np.argsort(totals))


def _bottleneck_initial(pt):
    n_jobs, n_machines = pt.shape
    machine_loads = pt.sum(axis=0)
    bottleneck = int(np.argmax(machine_loads))
    return list(np.argsort(-pt[:, bottleneck]))


def _mean_initial(pt):
    return list(np.argsort(-pt.mean(axis=1)))


def _first_machine_initial(pt):
    return list(np.argsort(-pt[:, 0]))


def _last_machine_initial(pt):
    return list(np.argsort(-pt[:, -1]))


def _random_initial(n_jobs, seed):
    rng = random.Random(seed)
    perm = list(range(n_jobs))
    rng.shuffle(perm)
    return perm


def _vnd_adjacent_swap(pt_list, perm, deadline):
    """N1: Adjacent swap neighborhood."""
    n = len(perm)
    if n <= 1:
        return perm, _makespan_fast(pt_list, perm)
    
    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    best = list(current)
    best_ms = current_ms
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                break
            test = current[:i] + [current[i+1]] + [current[i]] + current[i+2:]
            ms = _makespan_fast(pt_list, test)
            if ms < current_ms - 1e-9:
                current = test
                current_ms = ms
                improved = True
                if current_ms < best_ms - 1e-9:
                    best_ms = current_ms
                    best = list(current)
                break
    return best, best_ms


def _vnd_2opt(pt_list, perm, deadline):
    """N2: 2-opt segment reversal neighborhood."""
    n = len(perm)
    if n <= 2:
        return perm, _makespan_fast(pt_list, perm)
    
    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    best = list(current)
    best_ms = current_ms
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                test = current[:i] + current[i:j+1][::-1] + current[j+1:]
                ms = _makespan_fast(pt_list, test)
                if ms < current_ms - 1e-9:
                    current = test
                    current_ms = ms
                    improved = True
                    if current_ms < best_ms - 1e-9:
                        best_ms = current_ms
                        best = list(current)
                    break
            if improved:
                break
    return best, best_ms


def _vnd_or_opt(pt_list, perm, deadline):
    """N3: Or-opt block insertion neighborhood (blocks of 1-3)."""
    n = len(perm)
    if n <= 3:
        return perm, _makespan_fast(pt_list, perm)
    
    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    best = list(current)
    best_ms = current_ms
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for block_size in [1, 2, 3]:
            if time.time() > deadline:
                break
            for i in range(n - block_size + 1):
                if time.time() > deadline:
                    break
                block = current[i:i+block_size]
                rest = current[:i] + current[i+block_size:]
                
                for j in range(len(rest) + 1):
                    if time.time() > deadline:
                        break
                    if j == i:
                        continue
                    if j == i + 1:
                        continue
                    test = rest[:j] + block + rest[j:]
                    if len(test) != n:
                        continue
                    ms = _makespan_fast(pt_list, test)
                    if ms < current_ms - 1e-9:
                        current = test
                        current_ms = ms
                        improved = True
                        if current_ms < best_ms - 1e-9:
                            best_ms = current_ms
                            best = list(current)
                        break
                if improved:
                    break
            if improved:
                break
    return best, best_ms


def _vnd_local_search(pt_list, perm, deadline, max_cycles=20):
    """Variable Neighborhood Descent."""
    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    best = list(current)
    best_ms = current_ms
    
    neighborhoods = [
        _vnd_adjacent_swap,
        _vnd_2opt,
        _vnd_or_opt,
    ]
    
    for cycle in range(max_cycles):
        if time.time() > deadline:
            break
        improved_in_cycle = False
        for neighborhood in neighborhoods:
            if time.time() > deadline:
                break
            perm_new, ms_new = neighborhood(pt_list, current, deadline)
            if ms_new < current_ms - 1e-9:
                current = perm_new
                current_ms = ms_new
                improved_in_cycle = True
                if current_ms < best_ms - 1e-9:
                    best_ms = current_ms
                    best = list(current)
        if not improved_in_cycle:
            break
    
    return best, best_ms


def _discrete_pso(pt_list, pt, perm, deadline, max_iter=5000, seed=42):
    """
    Discrete Particle Swarm Optimization for permutation problems.
    Each particle has a position (permutation), personal best, and velocity
    represented as a list of (job, from_pos, to_pos) directives.
    """
    rng = random.Random(seed)
    n = len(perm)
    if n <= 1:
        return perm, _makespan_fast(pt_list, perm)
    
    # Initialize swarm
    pop_size = min(12, max(6, n // 3))
    
    # Initial population: mix of perturbed best and random
    population = [list(perm)]
    for _ in range(pop_size - 1):
        p = list(perm)
        # Random perturbation
        n_swaps = rng.randint(1, max(1, n // 4))
        for _ in range(n_swaps):
            a = rng.randint(0, n - 1)
            b = rng.randint(0, n - 1)
            p[a], p[b] = p[b], p[a]
        population.append(p)
    
    # Evaluate initial population
    fitnesses = [_makespan_fast(pt_list, p) for p in population]
    
    # Personal and global bests
    personal_bests = [list(p) for p in population]
    personal_fitnesses = list(fitnesses)
    global_best = list(population[0])
    global_fitness = fitnesses[0]
    for i in range(pop_size):
        if fitnesses[i] < global_fitness:
            global_fitness = fitnesses[i]
            global_best = list(population[i])
    
    # Velocity: for each particle, store a list of move directives
    # Each directive: (job_index, from_position, to_position)
    velocities = [[] for _ in range(pop_size)]
    
    w = 0.5  # inertia
    c1 = 1.5  # cognitive
    c2 = 1.5  # social
    w_decay = 0.999
    
    for _iter in range(max_iter):
        if time.time() > deadline:
            break
        
        # Update global best
        for i in range(pop_size):
            if fitnesses[i] < global_fitness:
                global_fitness = fitnesses[i]
                global_best = list(population[i])
        
        # Update each particle
        for i in range(pop_size):
            if time.time() > deadline:
                break
            
            # Update personal best
            if fitnesses[i] < personal_fitnesses[i]:
                personal_fitnesses[i] = fitnesses[i]
                personal_bests[i] = list(population[i])
            
            # Generate velocity directives
            # Cognitive: moves towards personal best
            # Social: moves towards global best
            new_velocity = []
            
            # Cognitive component
            if rng.random() < c1:
                # Find a job that is in a different position in personal best
                for pos in range(n):
                    if population[i][pos] != personal_bests[i][pos] and rng.random() < 0.3:
                        job = population[i][pos]
                        # Find where this job is in personal best
                        target_pos = personal_bests[i].index(job)
                        new_velocity.append((job, pos, target_pos))
            
            # Social component
            if rng.random() < c2:
                for pos in range(n):
                    if population[i][pos] != global_best[pos] and rng.random() < 0.3:
                        job = population[i][pos]
                        target_pos = global_best.index(job)
                        new_velocity.append((job, pos, target_pos))
            
            # Combine with previous velocity (with inertia)
            if velocities[i] and rng.random() < w:
                # Keep some old directives
                old_directives = velocities[i][:]
                rng.shuffle(old_directives)
                n_keep = max(1, len(old_directives) // 2)
                new_velocity = old_directives[:n_keep] + new_velocity
            else:
                new_velocity = new_velocity
            
            # Cap velocity
            max_velocity = max(3, n // 5)
            if len(new_velocity) > max_velocity:
                rng.shuffle(new_velocity)
                new_velocity = new_velocity[:max_velocity]
            
            velocities[i] = new_velocity
            
            # Apply velocity to update position
            new_perm = list(population[i])
            for (job, from_pos, to_pos) in new_velocity:
                # Check if job is still at from_pos
                if new_perm[from_pos] != job:
                    # Find where the job is now
                    try:
                        actual_from = new_perm.index(job)
                    except ValueError:
                        continue
                    from_pos = actual_from
                
                # Remove job from current position
                new_perm.pop(from_pos)
                
                # Adjust to_pos if needed
                if to_pos > from_pos:
                    to_pos -= 1
                to_pos = max(0, min(to_pos, len(new_perm)))
                
                # Insert job at new position
                new_perm.insert(to_pos, job)
            
            # Ensure valid permutation
            if len(new_perm) != n or set(new_perm) != set(range(n)):
                new_perm = list(population[i])
            
            population[i] = new_perm
            fitnesses[i] = _makespan_fast(pt_list, new_perm)
        
        w *= w_decay
        w = max(w, 0.1)
    
    # Return best found
    best_perm = global_best
    best_ms = global_fitness
    for i in range(pop_size):
        if fitnesses[i] < best_ms:
            best_ms = fitnesses[i]
            best_perm = list(population[i])
    
    return best_perm, best_ms


def _simulated_annealing(pt_list, perm, deadline, max_iter=100000, seed=0):
    """Simulated annealing with insertion moves."""
    rng = random.Random(seed)
    n = len(perm)
    if n <= 1:
        return perm, _makespan_fast(pt_list, perm)
    current = list(perm)
    current_ms = _makespan_fast(pt_list, current)
    best = list(current)
    best_ms = current_ms

    T = max(current_ms * 0.05, 1.0)
    T_min = max(current_ms * 0.0005, 0.01)
    cooling = 0.9995

    for _ in range(max_iter):
        if time.time() > deadline:
            break
        i = rng.randint(0, n - 1)
        job = current[i]
        rest = current[:i] + current[i+1:]
        j = rng.randint(0, n - 1)
        if j == i or j == i + 1:
            j = (j + 1) % n
        test = rest[:j] + [job] + rest[j:]
        new_ms = _makespan_fast(pt_list, test)
        delta = new_ms - current_ms
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current = test
            current_ms = new_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best = list(current)
        T *= cooling
        if T < T_min:
            T = T_min
    return best, best_ms


def order_jobs(processing_times):
    pt = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = pt.shape
    start = time.time()
    deadline = start + 145.0

    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    if n_machines == 0:
        return list(range(n_jobs))

    pt_list = pt.tolist()

    # Build diverse initial solutions
    candidates = [
        _neh_initial(pt),
        _palmer_initial(pt),
        _lpt_initial(pt),
        _spt_initial(pt),
        _bottleneck_initial(pt),
        _mean_initial(pt),
        _first_machine_initial(pt),
        _last_machine_initial(pt),
    ]
    # Add random starts
    for s in range(6):
        candidates.append(_random_initial(n_jobs, 100 + s))

    # Evaluate all, pick best
    best_ms = float('inf')
    best_perm = candidates[0]
    candidate_results = []
    for cand in candidates:
        ms = _makespan_fast(pt_list, cand)
        candidate_results.append((ms, cand))
        if ms < best_ms:
            best_ms = ms
            best_perm = cand

    # Sort candidates by makespan so we polish the best ones first
    candidate_results.sort(key=lambda x: x[0])

    # Multi-start local search: polish top candidates using VND
    for ms0, cand in candidate_results:
        if time.time() > deadline - 20.0:
            break
        # Skip if too far from best
        if ms0 > best_ms * 1.15:
            continue

        # VND local search
        perm, ms = _vnd_local_search(pt_list, cand, deadline - 20.0, max_cycles=10)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # Discrete PSO from best to explore broader space
    if time.time() < deadline - 25.0:
        perm_pso, ms_pso = _discrete_pso(pt_list, pt, best_perm, deadline - 25.0, max_iter=8000, seed=7)
        if ms_pso < best_ms:
            best_ms = ms_pso
            best_perm = perm_pso

    # Simulated annealing from best to escape local optima
    if time.time() < deadline - 15.0:
        perm, ms = _simulated_annealing(pt_list, best_perm, deadline - 15.0, max_iter=200000, seed=7)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # Final VND polish
    if time.time() < deadline - 3.0:
        perm, ms = _vnd_local_search(pt_list, best_perm, deadline - 2.0, max_cycles=15)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # Final adjacent swap polish (fast, fine-tuning)
    if time.time() < deadline - 1.0:
        perm_swap, ms_swap = _vnd_adjacent_swap(pt_list, best_perm, deadline - 0.5)
        if ms_swap < best_ms:
            best_ms = ms_swap
            best_perm = perm_swap

    return best_perm
# EVOLVE-BLOCK-END
