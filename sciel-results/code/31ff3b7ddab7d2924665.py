import numpy as np
import time

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    # Helper to calculate makespan for a given permutation
    def calc_makespan(perm):
        # perm is a list of job indices
        # We simulate the flow shop
        # completion[m] is the completion time on machine m
        completion = np.zeros(n_machines)
        for j in perm:
            job_times = processing_times[j]
            for m in range(n_machines):
                # The job can start on machine m only when:
                # 1. The machine is free (completion[m])
                # 2. The job has finished on the previous machine (completion[m-1] if m>0 else 0)
                if m == 0:
                    start_time = completion[0]
                else:
                    start_time = max(completion[m], completion[m-1])
                completion[m] = start_time + job_times[m]
        return completion[-1]

    # 1. Sort jobs by total processing time in descending order
    total_times = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-total_times)
    
    # 2. NEH Construction
    seq = [sorted_jobs[0]]
    
    for j in sorted_jobs[1:]:
        best_pos = 0
        best_makespan = float('inf')
        
        # Try inserting job j at each position in the current sequence
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [j] + seq[pos:]
            ms = calc_makespan(new_seq)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        seq.insert(best_pos, j)
        
        # Time check
        if time.time() > deadline:
            break

    # If time remains, we could do a local search, but NEH is already strong.
    # Let's add a simple local search if time permits to squeeze more out.
    # However, for a first jump, NEH is the key.
    
    return list(seq)
