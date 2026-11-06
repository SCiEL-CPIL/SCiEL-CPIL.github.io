import numpy as np
import time
import random

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 114.0  # safety margin before 150s kill
    
    n_jobs, n_machines = processing_times.shape
    
    # Helper to calculate makespan for a given permutation (list of job indices)
    def calc_makespan(perm):
        completion = np.zeros(n_machines)
        for j in perm:
            job_times = processing_times[j]
            for m in range(n_machines):
                if m == 0:
                    st = completion[0]
                else:
                    st = max(completion[m], completion[m-1])
                completion[m] = st + job_times[m]
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
    best_ms = calc_makespan(seq)

    # 2. Block-based decomposition
    # Partition jobs into blocks based on processing time correlations
    # Then optimize internal sequence of each block and block order separately
    
    # First, try to identify natural blocks by clustering jobs with similar processing profiles
    # Use a simple approach: sort by total processing time, then split into roughly equal blocks
    
    # Alternative: use the NEH sequence and try to identify contiguous segments that can be optimized independently
    
    # For now, let's try a different approach: use a more sophisticated local search
    # with multiple neighborhood structures and random restarts
    
    # 2. Multi-neighborhood local search with random restarts
    n_restarts = 5
    restarts_done = 0
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        # For restarts after the first, perturb the best solution found
        if restart == 0:
            current_seq = list(best_seq)
            current_ms = best_ms
        else:
            # Create a perturbed version of best_seq
            current_seq = list(best_seq)
            n_perturb = max(2, len(current_seq) // 5)
            for _ in range(n_perturb):
                i = random.randint(0, len(current_seq) - 2)
                j = random.randint(i + 1, len(current_seq) - 1)
                # Remove job at i, insert at j
                job = current_seq[i]
                current_seq = current_seq[:i] + current_seq[i+1:j+1] + [job] + current_seq[j+1:]
            current_ms = calc_makespan(current_seq)
        
        # Local search with multiple neighborhoods
        improved = True
        max_iterations = 10000
        iteration = 0
        
        while improved and time.time() < deadline - 10.0 and iteration < max_iterations:
            improved = False
            iteration += 1
            n = len(current_seq)
            
            # Neighborhood 1: Or-opt (insertion)
            for i in range(n - 1):
                if time.time() > deadline - 10.0:
                    break
                for j in range(i + 1, n):
                    job = current_seq[i]
                    new_seq = current_seq[:i] + current_seq[i+1:j+1] + [job] + current_seq[j+1:]
                    ms = calc_makespan(new_seq)
                    if ms < current_ms:
                        current_ms = ms
                        current_seq = new_seq
                        improved = True
                        if current_ms < best_ms:
                            best_ms = current_ms
                            best_seq = list(current_seq)
                        break
                if improved:
                    break
            
            if not improved:
                # Neighborhood 2: 2-opt (swap)
                for i in range(n - 1):
                    if time.time() > deadline - 10.0:
                        break
                    for j in range(i + 1, n):
                        new_seq = list(current_seq)
                        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                        ms = calc_makespan(new_seq)
                        if ms < current_ms:
                            current_ms = ms
                            current_seq = new_seq
                            improved = True
                            if current_ms < best_ms:
                                best_ms = current_ms
                                best_seq = list(current_seq)
                            break
                    if improved:
                        break
            
            if not improved:
                # Neighborhood 3: 3-opt (reverse segment)
                for i in range(n - 2):
                    if time.time() > deadline - 10.0:
                        break
                    for j in range(i + 1, n - 1):
                        new_seq = current_seq[:i+1] + current_seq[i+1:j+1][::-1] + current_seq[j+1:]
                        ms = calc_makespan(new_seq)
                        if ms < current_ms:
                            current_ms = ms
                            current_seq = new_seq
                            improved = True
                            if current_ms < best_ms:
                                best_ms = current_ms
                                best_seq = list(current_seq)
                            break
                    if improved:
                        break
        
        restarts_done += 1
    
    return best_seq
