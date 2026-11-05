<<<<<<< SEARCH
def _makespan_fast(processing_times, order):
    """Faster makespan computation using vectorized operations where possible."""
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    
    # Pre-extract the submatrix for the ordered jobs
    pt = processing_times[order]
    
    # For each machine, the completion time is the sum of processing times
    # plus the maximum of previous machines' completion times at each step.
    # We can compute this iteratively but vectorize across jobs where possible.
    
    # Simpler approach: for small n, just loop
    machine_time = np.zeros(n_machines)
    for j in range(n_jobs):
        for m in range(n_machines):
            prev = machine_time[m-1] if m > 0 else 0.0
            machine_time[m] = max(machine_time[m], prev) + pt[j, m]
    return machine_time[-1]
=======
def _makespan_fast(processing_times, order):
    """Faster makespan computation using vectorized operations where possible."""
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    
    pt = processing_times[order]
    machine_time = np.zeros(n_machines)
    for j in range(n_jobs):
        row = pt[j]
        for m in range(n_machines):
            prev = machine_time[m-1] if m > 0 else 0.0
            machine_time[m] = (machine_time[m] if machine_time[m] > prev else prev) + row[m]
    return machine_time[-1]
>>>>>>> REPLACE

