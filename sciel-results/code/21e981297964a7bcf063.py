"""Improved flow-shop scheduling via NEH heuristic with local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(processing_times, order):
    """Compute makespan for a given job order on a flow shop."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return 0.0
    if n_machines == 0:
        return 0.0
    
    # machine_completion[m] = when machine m becomes free (i.e., completion of last job on m)
    machine_completion = np.zeros(n_machines)
    
    for job_idx in order:
        # For this job, compute completion on each machine
        prev_completion = 0.0  # completion on previous machine for this job
        for m in range(n_machines):
            # Start time is max of when previous machine finished this job, and when this machine is free
            start_time = max(prev_completion, machine_completion[m])
            prev_completion = start_time + processing_times[job_idx, m]
            machine_completion[m] = prev_completion
    
    # Makespan is the maximum completion time across all machines
    return float(np.max(machine_completion)) if n_machines > 0 else 0.0


def _neh_heuristic(processing_times):
    """NEH heuristic for flow shop scheduling."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    # Sort jobs by total processing time in descending order
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    # Insert each job into the best position in the sequence
    sequence = [sorted_jobs[0]]
    
    for job in sorted_jobs[1:]:
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(sequence) + 1):
            candidate = sequence[:pos] + [job] + sequence[pos:]
            ms = _makespan(processing_times, candidate)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        sequence = sequence[:best_pos] + [job] + sequence[best_pos:]
    
    return sequence


def _palmer(processing_times):
    """Palmer's priority rule: sort by decreasing (n+1-j)*t_j1 - j*t_jn."""
    n_jobs, n_machines = processing_times.shape
    if n_machines == 0:
        return list(range(n_jobs))
    priorities = np.zeros(n_jobs)
    for j in range(n_jobs):
        priorities[j] = (n_jobs + 1 - j) * processing_times[j, 0] - j * processing_times[j, n_machines - 1]
    return list(np.argsort(-priorities))


def _local_search(processing_times, initial_order, time_limit=10.0):
    """Local search with swaps and Or-opt moves (1 and 2 job blocks)."""
    n_jobs = len(initial_order)
    if n_jobs <= 1:
        return list(initial_order)
    
    current_order = list(initial_order)
    current_makespan = _makespan(processing_times, current_order)
    best_order = list(current_order)
    best_makespan = current_makespan
    
    start_time = time.time()
    no_improvement_count = 0
    max_no_improvement = 3  # Allow a few passes without improvement before stopping
    
    while True:
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        pass_improvements = 0
        
        # 1. Try Swaps
        for i in range(n_jobs - 1):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n_jobs):
                if time.time() - start_time > time_limit:
                    break
                
                # Try swapping jobs at positions i and j
                new_order = current_order[:]
                new_order[i], new_order[j] = new_order[j], new_order[i]
                new_makespan = _makespan(processing_times, new_order)
                
                if new_makespan < current_makespan - 1e-9:
                    current_order = new_order
                    current_makespan = new_makespan
                    pass_improvements += 1
                    if current_makespan < best_makespan:
                        best_makespan = current_makespan
                        best_order = list(current_order)
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
            
        # 2. Try Or-opt moves (move 1 job)
        for i in range(n_jobs):
            if time.time() - start_time > time_limit:
                break
            for j in range(n_jobs):
                if i == j or (i > 0 and j == i - 1): # Skip trivial moves
                    continue
                if time.time() - start_time > time_limit:
                    break
                
                # Move job at i to position j
                job = current_order[i]
                new_order = current_order[:i] + current_order[i+1:]
                # Adjust j if removal shifted indices
                if i < j:
                    insert_pos = j - 1
                else:
                    insert_pos = j
                    
                if insert_pos < 0: insert_pos = 0
                if insert_pos > len(new_order): insert_pos = len(new_order)
                
                new_order = new_order[:insert_pos] + [job] + new_order[insert_pos:]
                
                new_makespan = _makespan(processing_times, new_order)
                
                if new_makespan < current_makespan - 1e-9:
                    current_order = new_order
                    current_makespan = new_makespan
                    pass_improvements += 1
                    if current_makespan < best_makespan:
                        best_makespan = current_makespan
                        best_order = list(current_order)
                    improved = True
                    break
            if improved:
                break

        if improved:
            continue

        # 3. Try Or-opt moves (move 2 job blocks)
        if n_jobs >= 2:
            for i in range(n_jobs - 1):
                if time.time() - start_time > time_limit:
                    break
                for j in range(n_jobs - 1):
                    if i == j or (i > 0 and j == i - 1) or (i < n_jobs-1 and j == i + 1):
                        continue
                    if time.time() - start_time > time_limit:
                        break
                    
                    # Move block of 2 jobs starting at i to position j
                    block = current_order[i:i+2]
                    new_order = current_order[:i] + current_order[i+2:]
                    
                    # Adjust j
                    if i < j:
                        insert_pos = j - 2
                    else:
                        insert_pos = j
                        
                    if insert_pos < 0: insert_pos = 0
                    if insert_pos > len(new_order): insert_pos = len(new_order)
                    
                    new_order = new_order[:insert_pos] + block + new_order[insert_pos:]
                    
                    new_makespan = _makespan(processing_times, new_order)
                    
                    if new_makespan < current_makespan - 1e-9:
                        current_order = new_order
                        current_makespan = new_makespan
                        pass_improvements += 1
                        if current_makespan < best_makespan:
                            best_makespan = current_makespan
                            best_order = list(current_order)
                        improved = True
                        break
                if improved:
                    break

        if not improved:
            no_improvement_count += 1
            if no_improvement_count >= max_no_improvement:
                break
        else:
            no_improvement_count = 0
    
    return best_order


def _random_perturbation(order, strength=0.3, rng=None):
    """Apply random perturbation with segment reversal and swaps."""
    n = len(order)
    if n <= 1:
        return list(order)
    
    if rng is None:
        rng = np.random
    
    new_order = list(order)
    num_moves = max(1, int(strength * n))
    for _ in range(num_moves):
        # Randomly choose between swap and segment reversal
        if rng.random() < 0.5:
            # Swap
            i, j = rng.choice(n, 2, replace=False)
            new_order[i], new_order[j] = new_order[j], new_order[i]
        else:
            # Segment reversal
            i = rng.randint(0, n)
            j = rng.randint(0, n)
            if i > j:
                i, j = j, i
            new_order[i:j+1] = new_order[i:j+1][::-1]
    
    return new_order


def _polish_pass(processing_times, initial_order, time_limit=5.0):
    """Final high-precision polish pass with tighter tolerance."""
    n_jobs = len(initial_order)
    if n_jobs <= 1:
        return list(initial_order)
    
    current_order = list(initial_order)
    current_makespan = _makespan(processing_times, current_order)
    best_order = list(current_order)
    best_makespan = current_makespan
    
    start_time = time.time()
    
    while True:
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        
        # Try all adjacent swaps with very tight tolerance
        for i in range(n_jobs - 1):
            if time.time() - start_time > time_limit:
                break
            new_order = current_order[:]
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            new_makespan = _makespan(processing_times, new_order)
            
            if new_makespan < current_makespan - 1e-12:
                current_order = new_order
                current_makespan = new_makespan
                if current_makespan < best_makespan:
                    best_makespan = current_makespan
                    best_order = list(current_order)
                improved = True
                break
        
        if improved:
            continue
        
        # Try all single job insertions
        for i in range(n_jobs):
            if time.time() - start_time > time_limit:
                break
            for j in range(n_jobs):
                if i == j or (i > 0 and j == i - 1):
                    continue
                if time.time() - start_time > time_limit:
                    break
                
                job = current_order[i]
                new_order = current_order[:i] + current_order[i+1:]
                if i < j:
                    insert_pos = j - 1
                else:
                    insert_pos = j
                    
                if insert_pos < 0: insert_pos = 0
                if insert_pos > len(new_order): insert_pos = len(new_order)
                
                new_order = new_order[:insert_pos] + [job] + new_order[insert_pos:]
                
                new_makespan = _makespan(processing_times, new_order)
                
                if new_makespan < current_makespan - 1e-12:
                    current_order = new_order
                    current_makespan = new_makespan
                    if current_makespan < best_makespan:
                        best_makespan = current_makespan
                        best_order = list(current_order)
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_order


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow shop.
    Uses NEH heuristic followed by local search with time management.
    """
    processing_times = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Deadline guard: we must return well before 150s
    deadline = 140.0  # 10s safety margin
    start_time = time.time()
    
    # Step 1: NEH heuristic
    neh_order = _neh_heuristic(processing_times)
    neh_makespan = _makespan(processing_times, neh_order)
    
    best_order = list(neh_order)
    best_makespan = neh_makespan
    
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    
    if remaining < 5.0:
        return best_order
    
    # Step 2: Local search on the NEH solution
    ls_time = min(remaining * 0.5, 60.0)  # Use up to half remaining time, max 60s
    ls_order = _local_search(processing_times, best_order, time_limit=ls_time)
    ls_makespan = _makespan(processing_times, ls_order)
    
    if ls_makespan < best_makespan:
        best_order = list(ls_order)
        best_makespan = ls_makespan
    
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    
    if remaining < 5.0:
        return best_order
    
    # Step 3: Multiple restarts with perturbation
    rng = np.random.RandomState(42)  # Reproducibility
    num_restarts = min(20, int(remaining / 2.0))  # Budget ~2s per restart
    
    for r in range(num_restarts):
        if time.time() + 2.0 > deadline:
            break
        
        # Perturb the best solution
        perturbed = _random_perturbation(best_order, strength=0.2 + 0.1 * r, rng=rng)
        
        # Local search on perturbed solution
        ls_time_r = min(1.5, deadline - time.time() - 1.0)
        if ls_time_r < 0.1:
            break
        
        ls_order_r = _local_search(processing_times, perturbed, time_limit=ls_time_r)
        ls_makespan_r = _makespan(processing_times, ls_order_r)
        
        if ls_makespan_r < best_makespan:
            best_order = list(ls_order_r)
            best_makespan = ls_makespan_r
    
    # Step 4: Final high-precision polish pass
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    
    if remaining > 3.0:
        polish_time = min(remaining - 2.0, 10.0)  # Leave 2s safety margin
        if polish_time > 1.0:
            polished_order = _polish_pass(processing_times, best_order, time_limit=polish_time)
            polished_makespan = _makespan(processing_times, polished_order)
            
            if polished_makespan < best_makespan:
                best_order = list(polished_order)
                best_makespan = polished_makespan
    
    return best_order
# EVOLVE-BLOCK-END
