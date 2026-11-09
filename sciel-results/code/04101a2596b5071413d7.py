<<<<<<< SEARCH
def _makespan(processing_times, order):
    """Compute makespan for a given job order efficiently using the recurrence."""
    n_jobs = len(order)
    if n_jobs == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    
    # C[i][j] = max(C[i-1][j], C[i][j-1]) + p[i][j]
    # We only need the previous row and current row.
    c_prev = np.zeros(n_machines, dtype=np.float64)
    
    for i in range(n_jobs):
        job_idx = order[i]
        p_row = processing_times[job_idx]
        c_curr = np.zeros(n_machines, dtype=np.float64)
        # c_curr[j] = max(c_prev[j], c_curr[j-1]) + p_row[j]
        # This is a sequential dependency along j, so we can't fully vectorize easily
        # without a scan, but for small M it's fast in Python or we can use a loop.
        # Let's use a simple loop for clarity and correctness, it's O(N*M).
        for j in range(n_machines):
            val = p_row[j]
            if j == 0:
                c_curr[j] = c_prev[j] + val
            else:
                c_curr[j] = max(c_prev[j], c_curr[j-1]) + val
        c_prev = c_curr
        
    return c_prev[-1]
=======
def _makespan(processing_times, order):
    """Compute makespan for a given job order efficiently using the recurrence."""
    n_jobs = len(order)
    if n_jobs == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    
    c_prev = np.zeros(n_machines, dtype=np.float64)
    
    for i in range(n_jobs):
        job_idx = order[i]
        p_row = processing_times[job_idx]
        c_curr = np.zeros(n_machines, dtype=np.float64)
        for j in range(n_machines):
            val = p_row[j]
            if j == 0:
                c_curr[j] = c_prev[j] + val
            else:
                c_curr[j] = max(c_prev[j], c_curr[j-1]) + val
        c_prev = c_curr
        
    return c_prev[-1]


def _delta_makespan_swap(processing_times, order, i, j):
    """
    Compute the change in makespan when swapping positions i and j in order.
    Uses the completion matrix approach for efficiency.
    """
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    if abs(i - j) == 1:
        # Adjacent swap - simpler computation
        a = order[i]
        b = order[i+1]
        
        # Compute completion times up to position i-1
        c = np.zeros(n_machines, dtype=np.float64)
        for k in range(i):
            p_row = processing_times[order[k]]
            for m in range(n_machines):
                if m == 0:
                    c[m] = c[m] + p_row[m]
                else:
                    c[m] = max(c[m], c[m-1]) + p_row[m]
        
        # Case 1: keep order [a, b]
        c1 = c.copy()
        for m in range(n_machines):
            if m == 0:
                c1[m] += processing_times[a, m]
            else:
                c1[m] = max(c1[m], c1[m-1]) + processing_times[a, m]
        for m in range(n_machines):
            if m == 0:
                c1[m] += processing_times[b, m]
            else:
                c1[m] = max(c1[m], c1[m-1]) + processing_times[b, m]
        
        # Case 2: swapped [b, a]
        c2 = c.copy()
        for m in range(n_machines):
            if m == 0:
                c2[m] += processing_times[b, m]
            else:
                c2[m] = max(c2[m], c2[m-1]) + processing_times[b, m]
        for m in range(n_machines):
            if m == 0:
                c2[m] += processing_times[a, m]
            else:
                c2[m] = max(c2[m], c2[m-1]) + processing_times[a, m]
        
        # Continue from position i+2 to end
        ms1 = c1[-1]
        ms2 = c2[-1]
        
        # Need to compute full makespan for both cases
        for k in range(i+2, n_jobs):
            p_row = processing_times[order[k]]
            for m in range(n_machines):
                if m == 0:
                    c1[m] += p_row[m]
                    c2[m] += p_row[m]
                else:
                    c1[m] = max(c1[m], c1[m-1]) + p_row[m]
                    c2[m] = max(c2[m], c2[m-1]) + p_row[m]
        
        return ms2 - ms1  # delta if we swap
    
    # For non-adjacent swaps, just recompute
    new_order = order.copy()
    new_order[i], new_order[j] = new_order[j], new_order[i]
    ms_orig = _makespan(processing_times, order)
    ms_new = _makespan(processing_times, new_order)
    return ms_new - ms_orig
>>>>>>> REPLACE

