"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(order, p):
    """
    Compute makespan for a given permutation of jobs.
    order: array of job indices
    p: processing times matrix (n_jobs, n_machines)
    
    Uses dynamic programming recurrence for flow shop:
    C[i, j] = max(C[i-1, j], C[i, j-1]) + p[i, j]
    where C is the completion time matrix.
    We only need the last element C[n-1, m-1].
    """
    n_jobs, n_machines = p.shape
    if n_jobs == 0:
        return 0.0
    
    # Start with the first job
    # cumsum over machines for the first job
    c_prev = np.cumsum(p[order[0]])
    
    # Iterate through remaining jobs
    for i in range(1, n_jobs):
        job_idx = order[i]
        # C[i, j] = max(C[i-1, j], C[i, j-1]) + p[i, j]
        # We have c_prev which is C[i-1, :]
        # We need to compute C[i, :]
        # C[i, 0] = c_prev[0] + p[i, 0]
        # C[i, j] = max(c_prev[j], C[i, j-1]) + p[i, j]
        
        # Vectorized approach for the row:
        # This is a sequential dependency within the row, so we can't fully vectorize 
        # the inner loop over machines easily without a scan, but n_machines is usually small.
        # However, we can use a simple loop which is fast enough for small M.
        # Or we can use a trick if M is small. Let's stick to a simple loop for clarity and correctness first.
        
        c_curr = np.empty(n_machines)
        c_curr[0] = c_prev[0] + p[job_idx, 0]
        for j in range(1, n_machines):
            c_curr[j] = max(c_prev[j], c_curr[j-1]) + p[job_idx, j]
        c_prev = c_curr
        
    return c_prev[-1]

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # 1. Initial Solution: NEH Heuristic
    total_times = processing_times.sum(axis=1)
    # Sort by decreasing total processing time
    order = list(np.argsort(-total_times))
    
    # NEH construction
    seq = np.array([order[0]])
    for job in order[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            # Insert job at pos
            new_seq = np.insert(seq, pos, job)
            ms = _makespan(new_seq, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = np.insert(seq, best_pos, job)
    
    current_order = seq
    current_ms = _makespan(current_order, processing_times)
    best_order = current_order.copy()
    best_ms = current_ms
    
    # 2. Simulated Annealing
    # Initial temperature based on problem size
    T = 100.0
    cooling_rate = 0.999
    min_T = 0.01
    
    # Precompute indices for random selection
    # We will perform random swaps
    
    while time.time() < deadline:
        if T < min_T:
            break
            
        # Generate a random neighbor
        i = np.random.randint(0, n_jobs - 1)
        j = np.random.randint(i + 1, n_jobs)
        
        # Swap jobs i and j
        new_order = current_order.copy()
        new_order[i], new_order[j] = new_order[j], new_order[i]
        
        new_ms = _makespan(new_order, processing_times)
        delta = new_ms - current_ms
        
        # Acceptance probability
        if delta < 0:
            # Better solution
            current_order = new_order
            current_ms = new_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best_order = current_order.copy()
        else:
            # Worse solution, accept with probability exp(-delta / T)
            if np.random.rand() < np.exp(-delta / T):
                current_order = new_order
                current_ms = new_ms
        
        # Cool down
        T *= cooling_rate
        
    return list(best_order)
# EVOLVE-BLOCK-END
