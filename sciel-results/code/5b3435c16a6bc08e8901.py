import numpy as np
import time

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 114.0  # safety margin before 150s kill
    
    n_jobs, n_machines = processing_times.shape
    
    # Helper to calculate makespan for a given permutation (list of job indices)
    def calc_makespan(perm):
        n = len(perm)
        if n == 0:
            return 0.0
        
        # Get processing times for all jobs in the permutation
        # Using np.array(perm) is fast enough for N < 200
        job_indices = np.array(perm, dtype=np.int64)
        job_times = processing_times[job_indices]
        
        # Compute completion times iteratively
        # prev_c holds completion times for previous job
        prev_c = np.zeros(n_machines)
        
        for j in range(n):
            jt = job_times[j]
            curr_c = np.zeros(n_machines)
            curr_c[0] = prev_c[0] + jt[0]
            for m in range(1, n_machines):
                # max(prev_c[m], curr_c[m-1])
                # Python max is faster than np.maximum for scalars
                curr_c[m] = max(prev_c[m], curr_c[m-1]) + jt[m]
            prev_c = curr_c
            
        return prev_c[-1]

    def neh_construction(sort_key):
        # sort_key is a function or array to sort by
        if isinstance(sort_key, np.ndarray):
            sorted_jobs = np.argsort(-sort_key)
        else:
            sorted_jobs = np.argsort(-sort_key(processing_times))
            
        seq = [sorted_jobs[0]]
        
        for j in sorted_jobs[1:]:
            best_pos = 0
            best_makespan = float('inf')
            
            for pos in range(len(seq) + 1):
                new_seq = seq[:pos] + [j] + seq[pos:]
                ms = calc_makespan(new_seq)
                if ms < best_makespan:
                    best_makespan = ms
                    best_pos = pos
            
            seq.insert(best_pos, j)
            
            if time.time() > deadline:
                break
                
        return seq

    def or_opt_search(seq, best_ms):
        n = len(seq)
        improved = True
        # Limit iterations to prevent infinite loops if no improvement is found
        # But we break if no improvement in a full pass
        max_passes = 100
        passes = 0
        
        while improved and time.time() < deadline and passes < max_passes:
            improved = False
            passes += 1
            n = len(seq)
            
            # Try block sizes 1, 2, 3
            for block_size in [1, 2, 3]:
                if time.time() > deadline:
                    return seq, best_ms
                if n < block_size:
                    continue
                    
                for i in range(n - block_size + 1):
                    if time.time() > deadline:
                        return seq, best_ms
                        
                    # Extract block
                    block = seq[i:i+block_size]
                    # Remove block
                    remaining = seq[:i] + seq[i+block_size:]
                    n_rem = len(remaining)
                    
                    # Try inserting at each position in remaining
                    for pos in range(n_rem + 1):
                        # Avoid inserting back at the same spot (rough check)
                        # If pos == i and we are moving from i, the resulting sequence is same
                        # But indices shift.
                        # If pos == i, new_seq is same as original?
                        # remaining[:i] + block + remaining[i:]
                        # Original: seq[:i] + block + seq[i+block_size:]
                        # Yes, if pos == i, it's the same sequence.
                        if pos == i:
                            continue
                            
                        new_seq = remaining[:pos] + block + remaining[pos:]
                        ms = calc_makespan(new_seq)
                        
                        if ms < best_ms:
                            best_ms = ms
                            seq = new_seq
                            n = len(seq)
                            improved = True
                            # Restart search from beginning of Or-opt
                            # Break out of all loops
                            break
                    if improved:
                        break
                if improved:
                    break
                    
        return seq, best_ms

    def two_opt_search(seq, best_ms):
        n = len(seq)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(seq)
            for i in range(n - 1):
                if time.time() > deadline:
                    return seq, best_ms
                for j in range(i + 1, n):
                    seq[i], seq[j] = seq[j], seq[i]
                    ms = calc_makespan(seq)
                    if ms < best_ms:
                        best_ms = ms
                        improved = True
                        # Restart 2-opt from beginning
                        break
                    else:
                        seq[i], seq[j] = seq[j], seq[i]
                if improved:
                    break
        return seq, best_ms

    # 1. Generate multiple initial solutions
    candidates = []
    
    # NEH with Sum
    candidates.append(neh_construction(lambda pt: pt.sum(axis=1)))
    
    # NEH with Mean
    candidates.append(neh_construction(lambda pt: pt.mean(axis=1)))
    
    # NEH with Max
    candidates.append(neh_construction(lambda pt: pt.max(axis=1)))
    
    # Random
    rng = np.random.default_rng(42)
    candidates.append(list(rng.permutation(n_jobs)))
    candidates.append(list(rng.permutation(n_jobs)))
    
    best_seq = candidates[0]
    best_ms = calc_makespan(best_seq)
    
    # 2. Apply Or-opt to each candidate
    for seq in candidates:
        if time.time() > deadline - 10: # Reserve time for final polish
            break
        ms = calc_makespan(seq)
        if ms < best_ms:
            best_ms = ms
            best_seq = list(seq)
            
        # Run Or-opt on this candidate
        seq, ms = or_opt_search(list(seq), ms)
        if ms < best_ms:
            best_ms = ms
            best_seq = list(seq)

    # 3. Final Polish with 2-opt on the best solution found
    if time.time() < deadline - 5:
        best_seq, best_ms = two_opt_search(list(best_seq), best_ms)

    return best_seq
