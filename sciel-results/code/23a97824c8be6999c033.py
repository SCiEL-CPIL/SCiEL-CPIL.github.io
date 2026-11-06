def _calc_makespan(processing_times, seq, n_machines, limit=None):
    completion_times = np.zeros(n_machines)
    pt = processing_times
    
    for job_idx in seq:
        for m in range(n_machines):
            if m == 0:
                completion_times[m] += pt[job_idx, m]
            else:
                completion_times[m] = max(completion_times[m], completion_times[m-1]) + pt[job_idx, m]
            
            if limit is not None and completion_times[m] > limit:
                return limit + 1.0
                
    return completion_times[-1]
