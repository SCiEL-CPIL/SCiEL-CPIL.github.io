import numpy as np
import time

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on local synthetic flow-shop instances.
    
    Strategy:
    1. Generate multiple initial solutions using NEH variants.
    2. Refine each solution using a fast local search with insertion moves.
    3. Return the best permutation found within the time budget.
    """
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Precompute total processing times for NEH sorting
    total_times = processing_times.sum(axis=1)
    
    # Helper function to calculate makespan for a given permutation
    def calc_makespan(perm):
        """
        Calculate the makespan for a given job permutation.
        perm: list of job indices
        """
        completion = np.zeros(n_machines)
        for job in perm:
            for m in range(n_machines):
                if m == 0:
                    completion[m] += processing_times[job, m]
                else:
                    completion[m] = max(completion[m], completion[m-1]) + processing_times[job, m]
        return completion[-1]
    
    def neh_heuristic(processing_times, total_times, rng_seed=None):
        """
        Implement the NEH heuristic.
        """
        n = len(total_times)
        # Sort jobs by decreasing total processing time
        # Add a small random noise to break ties and generate diverse starts
        if rng_seed is not None:
            rng = np.random.default_rng(rng_seed)
            noise = rng.uniform(0, 0.01, size=n)
            sort_keys = -(total_times + noise)
        else:
            sort_keys = -total_times
            
        sorted_jobs = np.argsort(sort_keys)
        
        # Build the sequence
        seq = [sorted_jobs[0]]
        
        for j in sorted_jobs[1:]:
            best_pos = 0
            best_makespan = float('inf')
            
            # Try inserting job j at every possible position in the current sequence
            for pos in range(len(seq) + 1):
                # Create a temporary sequence
                new_seq = seq[:pos] + [j] + seq[pos:]
                
                # Calculate makespan for this temporary sequence
                completion = np.zeros(n_machines)
                for k, job in enumerate(new_seq):
                    for m in range(n_machines):
                        if m == 0:
                            completion[m] += processing_times[job, m]
                        else:
                            completion[m] = max(completion[m], completion[m-1]) + processing_times[job, m]
                
                makespan = completion[-1]
                
                if makespan < best_makespan:
                    best_makespan = makespan
                    best_pos = pos
            
            # Insert job at best position
            seq.insert(best_pos, j)
            
        return seq, best_makespan
    
    def local_search_insertion(perm, initial_makespan, max_iterations, start_time_ref, time_limit_ref):
        """
        Perform local search using insertion moves.
        Returns the best permutation and its makespan.
        """
        n = len(perm)
        best_perm = perm[:]
        best_makespan = initial_makespan
        current_perm = perm[:]
        current_makespan = initial_makespan
        
        # Precompute completion times for the current permutation to speed up evaluation
        def get_completion_matrix(p):
            comp = np.zeros((n, n_machines))
            for idx, job in enumerate(p):
                for m in range(n_machines):
                    if m == 0:
                        comp[idx, m] = processing_times[job, m]
                    else:
                        comp[idx, m] = max(comp[idx-1, m], comp[idx, m-1]) + processing_times[job, m]
            return comp
        
        current_comp = get_completion_matrix(current_perm)
        
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iterations:
            improved = False
            iter_count += 1
            
            if time.time() - start_time_ref > time_limit_ref:
                break
                
            for i in range(n):
                if time.time() - start_time_ref > time_limit_ref:
                    break
                
                job = current_perm[i]
                # Remove job from position i
                new_perm_base = current_perm[:i] + current_perm[i+1:]
                
                for j in range(n):
                    if i == j:
                        continue
                    
                    # Determine the target position in the new list (after removal)
                    if j < i:
                        insert_idx = j
                    else:
                        insert_idx = j - 1
                    
                    # Clamp insert_idx
                    if insert_idx < 0: insert_idx = 0
                    if insert_idx > len(new_perm_base): insert_idx = len(new_perm_base)
                    
                    new_perm = new_perm_base[:insert_idx] + [job] + new_perm_base[insert_idx:]
                    
                    # Calculate makespan for new_perm
                    completion = np.zeros(n_machines)
                    for jb in new_perm:
                        for m in range(n_machines):
                            if m == 0:
                                completion[m] += processing_times[jb, m]
                            else:
                                completion[m] = max(completion[m], completion[m-1]) + processing_times[jb, m]
                    new_makespan = completion[-1]
                    
                    if new_makespan < current_makespan:
                        current_perm = new_perm
                        current_makespan = new_makespan
                        if current_makespan < best_makespan:
                            best_makespan = current_makespan
                            best_perm = current_perm[:]
                        improved = True
                        break # Break inner loop, restart outer? No, just break to next i?
        
        return best_perm, best_makespan

    # Generate multiple diverse NEH starts
    num_starts = 15
    best_overall_perm = None
    best_overall_makespan = float('inf')
    
    for i in range(num_starts):
        if time.time() - start_time > time_limit:
            break
            
        # Use different seeds for diversity
        perm, makespan = neh_heuristic(processing_times, total_times, rng_seed=i)
        
        if makespan < best_overall_makespan:
            best_overall_makespan = makespan
            best_overall_perm = perm[:]
            
        # Refine with local search
        # Dynamic iteration limit based on remaining time
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        # Assume ~0.001s per iteration (conservative for Python)
        max_iters = int(remaining * 500) 
        if max_iters < 50:
            max_iters = 50
        if max_iters > 2000:
            max_iters = 2000
            
        refined_perm, refined_makespan = local_search_insertion(
            perm, makespan, max_iters, start_time, time_limit
        )
        
        if refined_makespan < best_overall_makespan:
            best_overall_makespan = refined_makespan
            best_overall_perm = refined_perm[:]
            
    # If no best found (should not happen), fallback to NEH
    if best_overall_perm is None:
        best_overall_perm, _ = neh_heuristic(processing_times, total_times)
        
    return best_overall_perm
