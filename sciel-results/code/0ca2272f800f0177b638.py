import numpy as np
import time

def _makespan(processing_times, order):
    """Compute makespan for a given job order."""
    if len(order) == 0:
        return 0.0
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    # C[j] = completion time of current job on machine j
    # We need to track the completion time of the previous job on each machine
    # to calculate max(C_prev[j], C_curr[j-1]) + p
    
    # completion[j] holds the time when machine j is free
    # job_completion holds the time when the current job finishes on machine j-1
    completion = np.zeros(n_machines)
    
    for job_idx in order:
        job_completion = 0.0
        for j in range(n_machines):
            # Machine j starts at max(time it was free, time job finished on prev machine)
            start = max(completion[j], job_completion)
            job_completion = start + processing_times[job_idx, j]
            completion[j] = job_completion
        # job_completion is now the completion time on the last machine
    return completion[-1]

def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Initial solution: NEH heuristic
    total_pt = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-total_pt)
    
    order = []
    # First job
    order.append(sorted_jobs[0])
    
    # Second job: try both orders
    if n_jobs > 1:
        j1, j2 = sorted_jobs[0], sorted_jobs[1]
        ms_12 = _makespan(processing_times, [j1, j2])
        ms_21 = _makespan(processing_times, [j2, j1])
        if ms_12 <= ms_21:
            order = [j1, j2]
        else:
            order = [j2, j1]
    else:
        order = [sorted_jobs[0]]
        
    # Insert rest
    for k in range(2, n_jobs):
        job = sorted_jobs[k]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(order) + 1):
            cand = order[:pos] + [job] + order[pos:]
            ms = _makespan(processing_times, cand)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        order = order[:best_pos] + [job] + order[best_pos:]
        
    best_order = order
    best_ms = _makespan(processing_times, best_order)
    
    # Simulated Annealing
    rng = np.random.RandomState(42)
    start_time = time.time()
    time_limit = 100 # Leave buffer for overhead
    
    T = best_ms * 0.5
    T_min = 0.01
    cooling_rate = 0.995
    
    current_order = best_order
    current_ms = best_ms
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
            
        # Generate neighbor
        # Choose between swap and insertion
        if rng.random() < 0.5:
            # Swap two random positions
            i, j = rng.choice(n_jobs, 2, replace=False)
            neighbor = current_order.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        else:
            # Insertion move
            i, j = rng.choice(n_jobs, 2, replace=False)
            neighbor = current_order.copy()
            job = neighbor.pop(i)
            # Adjust j if i < j
            if i < j:
                j -= 1
            neighbor.insert(j, job)
            
        neighbor_ms = _makespan(processing_times, neighbor)
        delta = neighbor_ms - current_ms
        
        # Acceptance probability
        if delta < 0:
            # Better, always accept
            current_order = neighbor
            current_ms = neighbor_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best_order = current_order.copy()
        else:
            # Worse, accept with probability exp(-delta/T)
            prob = np.exp(-delta / T)
            if rng.random() < prob:
                current_order = neighbor
                current_ms = neighbor_ms
                
        T *= cooling_rate
        
    return best_order
