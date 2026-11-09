"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment


def _makespan(processing_times, order):
    """Compute makespan for a given job order on a flow shop."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return 0.0
    # completion[j][m] = completion time of job j on machine m
    # We can compute this efficiently
    completion = np.zeros((n_jobs, n_machines), dtype=np.float64)
    for j_idx, job in enumerate(order):
        for m in range(n_machines):
            if m == 0:
                completion[j_idx, m] = (completion[j_idx - 1, m] if j_idx > 0 else 0.0) + processing_times[job, m]
            else:
                # max of previous machine for same job, and previous job on same machine
                prev_machine = completion[j_idx, m - 1]
                prev_job = completion[j_idx - 1, m] if j_idx > 0 else 0.0
                completion[j_idx, m] = max(prev_machine, prev_job) + processing_times[job, m]
    return completion[-1, -1]


def _makespan_fast(processing_times, order):
    """Faster makespan computation using numpy where possible."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return 0.0
    # Build the submatrix for the given order
    ordered = processing_times[order]  # shape (n_jobs, n_machines)
    
    # DP: C[i][j] = completion of job i on machine j
    # C[i][j] = max(C[i-1][j], C[i][j-1]) + p[i][j]
    # This is O(n*m) but with numpy we can do it iteratively
    
    # For small instances, the simple loop is fine
    if n_jobs * n_machines < 10000:
        completion = np.zeros((n_jobs, n_machines), dtype=np.float64)
        for j_idx in range(n_jobs):
            for m in range(n_machines):
                if m == 0:
                    completion[j_idx, m] = (completion[j_idx - 1, m] if j_idx > 0 else 0.0) + ordered[j_idx, m]
                else:
                    completion[j_idx, m] = max(completion[j_idx, m - 1], 
                                               completion[j_idx - 1, m] if j_idx > 0 else 0.0) + ordered[j_idx, m]
        return completion[-1, -1]
    
    # For larger instances, still use the loop but it should be okay
    completion = np.zeros((n_jobs, n_machines), dtype=np.float64)
    for j_idx in range(n_jobs):
        for m in range(n_machines):
            if m == 0:
                completion[j_idx, m] = (completion[j_idx - 1, m] if j_idx > 0 else 0.0) + ordered[j_idx, m]
            else:
                completion[j_idx, m] = max(completion[j_idx, m - 1], 
                                           completion[j_idx - 1, m] if j_idx > 0 else 0.0) + ordered[j_idx, m]
    return completion[-1, -1]


def _neh_heuristic(processing_times):
    """NEH heuristic for flow shop scheduling."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return list(range(n_jobs))
    
    # Sort jobs by decreasing total processing time
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    # Insert each job at the best position
    order = [sorted_jobs[0]]
    for job in sorted_jobs[1:]:
        best_pos = 0
        best_ms = _makespan_fast(processing_times, order + [job])
        for pos in range(1, len(order) + 1):
            candidate = order[:pos] + [job] + order[pos:]
            ms = _makespan_fast(processing_times, candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        order = order[:best_pos] + [job] + order[best_pos:]
    
    return order


def _dispatching_rule(processing_times, rule='lpt'):
    """Simple dispatching rule."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return list(range(n_jobs))
    
    if rule == 'lpt':
        return list(np.argsort(-processing_times.sum(axis=1)))
    elif rule == 'spt':
        return list(np.argsort(processing_times.sum(axis=1)))
    elif rule == 'gws':
        # Greatest Weighted Sum: sum of p[i][j] * (n_machines - j)
        weights = np.arange(n_machines, 0, -1)
        scores = processing_times @ weights
        return list(np.argsort(-scores))
    else:
        return list(range(n_jobs))


def _local_search(processing_times, order, max_iterations=100, time_limit=None):
    """Local search using Or-opt (block) moves."""
    n_jobs = len(order)
    if n_jobs <= 1:
        return order, _makespan_fast(processing_times, order)
    
    best_order = list(order)
    best_ms = _makespan_fast(processing_times, best_order)
    
    start_time = time.time()
    no_improve = 0
    max_no_improve = 20
    
    for iteration in range(max_iterations):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        if no_improve >= max_no_improve:
            break
        
        found_improvement = False
        
        # Try Or-opt moves: remove a block of size 1, 2, or 3 and reinsert
        for block_size in [1, 2, 3]:
            for i in range(n_jobs - block_size + 1):
                # Extract block
                block = best_order[i:i+block_size]
                remaining = best_order[:i] + best_order[i+block_size:]
                
                # Try all insertion positions
                for j in range(len(remaining) - block_size + 1):
                    if j == i:
                        continue
                    candidate = remaining[:j] + block + remaining[j:]
                    ms = _makespan_fast(processing_times, candidate)
                    if ms < best_ms - 1e-9:
                        best_order = candidate
                        best_ms = ms
                        found_improvement = True
                        no_improve = 0
                        break
                if found_improvement:
                    break
            if found_improvement:
                break
        
        if not found_improvement:
            no_improve += 1
    
    return best_order, best_ms


def _discrete_pso(processing_times, n_particles, max_iterations, time_limit):
    """Discrete Particle Swarm Optimization for flow shop."""
    n_jobs = processing_times.shape[0]
    if n_jobs == 0:
        return [], 0.0
    
    start_time = time.time()
    
    # Initialize particles with diverse heuristics
    particles = []
    for i in range(n_particles):
        if i == 0:
            p = _neh_heuristic(processing_times)
        elif i == 1:
            p = _dispatching_rule(processing_times, 'lpt')
        elif i == 2:
            p = _dispatching_rule(processing_times, 'spt')
        elif i == 3:
            p = _dispatching_rule(processing_times, 'gws')
        elif i == 4:
            p = list(range(n_jobs))
        elif i == 5:
            p = list(range(n_jobs - 1, -1, -1))
        else:
            p = list(np.random.permutation(n_jobs))
        particles.append(p)
    
    # Evaluate initial particles
    particle_fits = []
    for p in particles:
        ms = _makespan_fast(processing_times, p)
        particle_fits.append(ms)
    
    # Personal bests
    personal_best = [list(p) for p in particles]
    personal_best_fit = list(particle_fits)
    
    # Global best
    global_best_idx = int(np.argmin(personal_best_fit))
    global_best = list(personal_best[global_best_idx])
    global_best_fit = personal_best_fit[global_best_idx]
    
    # Run local search on global best
    global_best, global_best_fit = _local_search(processing_times, global_best, 
                                                   max_iterations=50, 
                                                   time_limit=min(5.0, time_limit * 0.1))
    
    for iteration in range(max_iterations):
        if (time.time() - start_time) > time_limit:
            break
        
        for i in range(n_particles):
            if (time.time() - start_time) > time_limit:
                    break
            
            # Discrete velocity: blend personal and global best
            # Use a random crossover-like operation
            p_best = personal_best[i]
            g_best = global_best
            
            # Create a new particle by combining p_best and g_best
            # For each position, randomly choose from p_best or g_best
            new_particle = []
            used = set()
            for pos in range(n_jobs):
                if random.random() < 0.5:
                    # Take from personal best
                    for job in p_best:
                        if job not in used:
                            new_particle.append(job)
                            used.add(job)
                            break
                else:
                    # Take from global best
                    for job in g_best:
                        if job not in used:
                            new_particle.append(job)
                            used.add(job)
                            break
            
            # Add any remaining jobs
            for job in range(n_jobs):
                if job not in used:
                    new_particle.append(job)
            
            # Evaluate
            new_fit = _makespan_fast(processing_times, new_particle)
            
            # Update personal best
            if new_fit < personal_best_fit[i]:
                personal_best[i] = list(new_particle)
                personal_best_fit[i] = new_fit
                
                # Update global best
                if new_fit < global_best_fit:
                    global_best = list(new_particle)
                    global_best_fit = new_fit
            
            # Also try the new particle itself
            if new_fit < global_best_fit:
                global_best = list(new_particle)
                global_best_fit = new_fit
    
    # Final local search on global best
    global_best, global_best_fit = _local_search(processing_times, global_best,
                                                   max_iterations=100,
                                                   time_limit=min(10.0, time_limit * 0.2))
    
    return global_best, global_best_fit


def _simulated_annealing(processing_times, order, time_limit):
    """Simulated annealing for flow shop."""
    n_jobs = len(order)
    if n_jobs <= 1:
        return order, _makespan_fast(processing_times, order)
    
    start_time = time.time()
    current_order = list(order)
    current_ms = _makespan_fast(processing_times, current_order)
    best_order = list(current_order)
    best_ms = current_ms
    
    T = 100.0
    T_min = 0.1
    cooling_rate = 0.99
    
    while (time.time() - start_time) < time_limit and T > T_min:
        # Generate neighbor: swap two jobs or Or-opt move
        if random.random() < 0.5:
            # Swap
            i, j = random.sample(range(n_jobs), 2)
            neighbor = list(current_order)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        else:
            # Or-opt
            block_size = random.randint(1, min(3, n_jobs))
            i = random.randint(0, n_jobs - block_size)
            block = current_order[i:i+block_size]
            remaining = current_order[:i] + current_order[i+block_size:]
            j = random.randint(0, len(remaining) - block_size)
            neighbor = remaining[:j] + block + remaining[j:]
        
        neighbor_ms = _makespan_fast(processing_times, neighbor)
        delta = neighbor_ms - current_ms
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_order = neighbor
            current_ms = neighbor_ms
            
            if current_ms < best_ms:
                best_order = list(current_order)
                best_ms = current_ms
        
        T *= cooling_rate
    
    return best_order, best_ms


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    
    Uses a hybrid approach:
    1. Multiple heuristic initializations (NEH, dispatching rules)
    2. Discrete PSO for exploration
    3. Local search with Or-opt moves for exploitation
    4. Simulated annealing for fine-tuning
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    processing_times = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Small instance: exact-ish search
    if n_jobs <= 8:
        # Use NEH + thorough local search
        best_order = _neh_heuristic(processing_times)
        best_ms = _makespan_fast(processing_times, best_order)
        
        # Try multiple starts
        for rule in ['lpt', 'spt', 'gws']:
            order = _dispatching_rule(processing_times, rule)
            ms = _makespan_fast(processing_times, order)
            if ms < best_ms:
                best_ms = ms
                best_order = order
        
        # Local search
        best_order, best_ms = _local_search(processing_times, best_order, 
                                             max_iterations=500,
                                             time_limit=10.0)
        
        # Simulated annealing
        best_order, best_ms = _simulated_annealing(processing_times, best_order, 
                                                    time_limit=10.0)
        
        return best_order
    
    # Medium instance: hybrid approach
    if n_jobs <= 50:
        # Phase 1: Generate diverse initial solutions
        initial_orders = []
        initial_orders.append(_neh_heuristic(processing_times))
        initial_orders.append(_dispatching_rule(processing_times, 'lpt'))
        initial_orders.append(_dispatching_rule(processing_times, 'spt'))
        initial_orders.append(_dispatching_rule(processing_times, 'gws'))
        
        # Add some random permutations
        for _ in range(4):
            initial_orders.append(list(np.random.permutation(n_jobs)))
        
        # Phase 2: Local search on each
        best_order = None
        best_ms = float('inf')
        
        for order in initial_orders:
            if (time.time() - start_time) > deadline:
                break
            searched_order, ms = _local_search(processing_times, order,
                                                max_iterations=100,
                                                time_limit=2.0)
            if ms < best_ms:
                best_ms = ms
                best_order = searched_order
        
        # Phase 3: PSO on the best
        if (time.time() - start_time) < deadline - 10.0:
            pso_order, pso_ms = _discrete_pso(processing_times, 
                                               n_particles=10,
                                               max_iterations=50,
                                               time_limit=min(15.0, deadline - (time.time() - start_time) - 5.0))
            if pso_ms < best_ms:
                best_ms = pso_ms
                best_order = pso_order
        
        # Phase 4: Final local search + SA
        if (time.time() - start_time) < deadline - 5.0:
            best_order, best_ms = _local_search(processing_times, best_order,
                                                 max_iterations=200,
                                                 time_limit=min(5.0, deadline - (time.time() - start_time) - 2.0))
        
        if (time.time() - start_time) < deadline - 2.0:
            best_order, best_ms = _simulated_annealing(processing_times, best_order,
                                                        time_limit=min(3.0, deadline - (time.time() - start_time)))
        
        return best_order
    
    # Large instance: focus on fast heuristics + limited local search
    # NEH is the best heuristic for large instances
    best_order = _neh_heuristic(processing_times)
    best_ms = _makespan_fast(processing_times, best_order)
    
    # Try a few dispatching rules
    for rule in ['lpt', 'spt', 'gws']:
        order = _dispatching_rule(processing_times, rule)
        ms = _makespan_fast(processing_times, order)
        if ms < best_ms:
            best_ms = ms
            best_order = order
    
    # Limited local search
    if (time.time() - start_time) < deadline - 5.0:
        best_order, best_ms = _local_search(processing_times, best_order,
                                             max_iterations=50,
                                             time_limit=min(10.0, deadline - (time.time() - start_time) - 2.0))
    
    # Limited SA
    if (time.time() - start_time) < deadline - 2.0:
        best_order, best_ms = _simulated_annealing(processing_times, best_order,
                                                    time_limit=min(5.0, deadline - (time.time() - start_time)))
    
    return best_order
# EVOLVE-BLOCK-END
