import numpy as np
import time

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on local synthetic flow-shop instances.
    """
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Precompute total processing times for NEH
    total_times = processing_times.sum(axis=1)
    
    def calc_makespan(perm):
        """
        Calculate makespan for a given permutation using vectorized NumPy operations.
        """
        if n_jobs == 0:
            return 0.0
        
        # prev_comp holds completion times for the previous job
        prev_comp = np.zeros(n_machines)
        
        # Process first job
        job = perm[0]
        prev_comp = processing_times[job, :].copy()
        
        # Process remaining jobs
        for idx in range(1, n_jobs):
            job = perm[idx]
            pt = processing_times[job, :]
            
            curr_comp = np.empty(n_machines)
            curr_comp[0] = prev_comp[0] + pt[0]
            
            if n_machines > 1:
                # Vectorized update for machines 1..M-1
                # C[j, m] = max(C[j-1, m], C[j, m-1]) + P[j, m]
                curr_comp[1:] = np.maximum(prev_comp[1:], curr_comp[:-1]) + pt[1:]
                
            prev_comp = curr_comp
            
        return prev_comp[-1]

    def neh_heuristic(rng):
        """
        NEH Heuristic with randomization.
        """
        n = n_jobs
        # Sort by total processing time, with noise
        noise = rng.uniform(0, 1.0, size=n) * 0.01
        sort_keys = -(total_times + noise)
        sorted_jobs = np.argsort(sort_keys)
        
        seq = [sorted_jobs[0]]
        
        for job in sorted_jobs[1:]:
            best_pos = 0
            best_ms = float('inf')
            
            # Try inserting at each position
            for pos in range(len(seq) + 1):
                # Create temp seq
                temp_seq = seq[:pos] + [job] + seq[pos:]
                ms = calc_makespan(temp_seq)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            
            seq.insert(best_pos, job)
            
        return seq, calc_makespan(seq)

    def local_search(perm, current_ms, time_budget):
        """
        Local search with insertion moves.
        First improvement strategy.
        """
        n = len(perm)
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > time_budget:
                break
                
            for i in range(n):
                if time.time() - start_time > time_budget:
                    break
                job = perm[i]
                
                # Try inserting at all other positions
                # We skip i and i-1 (which results in the same permutation)
                for j in range(n):
                    if j == i or j == i - 1:
                        continue
                        
                    # Construct new perm
                    base = perm[:i] + perm[i+1:]
                    insert_idx = j if j < i else j - 1
                    if insert_idx < 0: insert_idx = 0
                    if insert_idx > len(base): insert_idx = len(base)
                    
                    new_perm = base[:insert_idx] + [job] + base[insert_idx:]
                    new_ms = calc_makespan(new_perm)
                    
                    if new_ms < current_ms:
                        perm = new_perm
                        current_ms = new_ms
                        improved = True
                        break
                if improved:
                    break
                    
        return perm, current_ms

    def perturb(perm, rng, strength=2):
        """
        Random swap perturbation.
        """
        perm = perm[:]
        n = len(perm)
        for _ in range(strength):
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if i == j:
                continue
            perm[i], perm[j] = perm[j], perm[i]
        return perm

    best_perm = None
    best_ms = float('inf')
    rng = np.random.default_rng(42)
    
    # Phase 1: Multiple NEH starts
    num_starts = 5
    for _ in range(num_starts):
        if time.time() - start_time > time_limit:
            break
        perm, ms = neh_heuristic(rng)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm[:]
            
    # Phase 2: Local Search on best
    if best_perm is not None:
        best_perm, best_ms = local_search(best_perm, best_ms, time_limit)
        
    # Phase 3: Iterated Local Search
    while time.time() - start_time < time_limit:
        # Perturb
        perturbed = perturb(best_perm, rng, strength=2)
        
        # Run Local Search
        # We pass the global deadline to local_search
        new_perm, new_ms = local_search(perturbed, calc_makespan(perturbed), time_limit)
        
        if new_ms < best_ms:
            best_ms = new_ms
            best_perm = new_perm[:]
            
    return best_perm
