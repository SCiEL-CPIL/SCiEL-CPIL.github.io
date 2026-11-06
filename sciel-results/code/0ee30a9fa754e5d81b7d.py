import numpy as np
import time

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 114.0  # safety margin before 150s kill
    
    n_jobs, n_machines = processing_times.shape
    
    # Helper to calculate makespan for a given permutation (list of job indices)
    def calc_makespan(perm):
        # perm is a list of job indices
        # Use vectorized numpy operations for speed
        n = len(perm)
        if n == 0:
            return 0.0
        
        # Get processing times for all jobs in the permutation
        job_indices = np.array(perm, dtype=np.int64)
        job_times = processing_times[job_indices]  # shape: (n_jobs, n_machines)
        
        # Compute completion times iteratively (can't fully vectorize due to dependencies)
        completion = np.zeros(n_machines)
        for j in range(n):
            jt = job_times[j]
            for m in range(n_machines):
                if m == 0:
                    st = completion[0]
                else:
                    st = max(completion[m], completion[m-1])
                completion[m] = st + jt[m]
        return completion[-1]

    # 1. NEH Construction
    # Sort jobs by total processing time in descending order
    total_times = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-total_times)
    
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
            return list(seq)

    best_seq = list(seq)
    best_ms = best_makespan

    # 2. Local Search: 2-opt / Swap Neighborhood
    # Try swapping pairs of jobs in the sequence
    # We use a simple hill climbing approach
    # To avoid O(n^3) per iteration being too slow, we limit iterations or use random sampling if n is large
    # However, for typical flow shop sizes (n < 100), full 2-opt might be feasible within time.
    # Let's do a standard 2-opt search (swap any two jobs).
    
    # Precompute for faster makespan updates
    # For swap-based local search, we can compute the change in makespan more efficiently
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(seq)
        # Iterate over all pairs (i, j) with i < j
        for i in range(n - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                # Swap seq[i] and seq[j]
                seq[i], seq[j] = seq[j], seq[i]
                ms = calc_makespan(seq)
                if ms < best_ms:
                    best_ms = ms
                    best_seq = list(seq)
                    improved = True
                else:
                    # Revert swap
                    seq[i], seq[j] = seq[j], seq[i]
    
    # 3. Additional restarts with different initial solutions if time remains
    if time.time() < deadline - 30:  # Need at least 30s for final polish
        # Try a few different random restarts
        rng = np.random.default_rng(42)
        for restart in range(3):
            if time.time() > deadline - 30:
                break
            
            # Random initial permutation
            seq = list(rng.permutation(n_jobs))
            ms = calc_makespan(seq)
            
            if ms < best_ms:
                best_ms = ms
                best_seq = list(seq)
            
            # Quick local search on this restart
            local_improved = True
            while local_improved and time.time() < deadline - 15:
                local_improved = False
                n = len(seq)
                for i in range(n - 1):
                    if time.time() > deadline - 15:
                        break
                    for j in range(i + 1, n):
                        seq[i], seq[j] = seq[j], seq[i]
                        ms = calc_makespan(seq)
                        if ms < best_ms:
                            best_ms = ms
                            best_seq = list(seq)
                            local_improved = True
                        else:
                            seq[i], seq[j] = seq[j], seq[i]
    
    return best_seq
