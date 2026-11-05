# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using Multi-Start Iterated Local Search (ILS).
    Uses NEH for initial construction and 2-opt (segment reversal) for local search.
    Vectorized makespan evaluation allows for many restarts within the time limit.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Fast fallback for very large instances where local search is too slow
    if n_jobs * n_machines > 500000:
        # Use a strong heuristic: LPT (Longest Processing Time)
        totals = processing_times.sum(axis=1)
        return list(np.argsort(-totals))
    
    def makespan(seq):
        """
        Compute makespan for a given sequence of job indices.
        Vectorized across jobs using NumPy operations where possible, 
        but the flow-shop dependency requires a loop over jobs.
        """
        if len(seq) == 0:
            return 0.0
        
        # completion[m] is the completion time of the current job on machine m
        # We process jobs one by one.
        # To speed up, we can use numpy arrays for the completion state.
        
        # Initialize completion times for the first job
        # completion[job_idx, m] is not directly useful for incremental, 
        # but for a full makespan, we just track the 'front' of the flow shop.
        
        # Let's use a simple loop which is often fast enough for n<1000, 
        # but we can optimize the inner loop.
        
        # Actually, a fully vectorized makespan is hard due to the max dependency.
        # However, we can precompute the processing times matrix.
        
        # Standard flow shop makespan calculation:
        # C[j][m] = max(C[j][m-1], C[j-1][m]) + P[j][m]
        
        # We only need the last row (last job) and last column (last machine).
        # But we need the previous row to compute the current one.
        
        # Let's stick to a Python loop over jobs, but use numpy for the machine loop?
        # No, the machine loop is also sequential.
        
        # Optimization: Pre-fetch job rows.
        # seq is a list of indices.
        
        # We will compute the completion time vector for each job.
        # C_prev is the completion vector of the previous job (shape: n_machines)
        # C_curr is the completion vector of the current job.
        
        # For the first job:
        # C_curr[0] = P[job, 0]
        # C_curr[m] = C_curr[m-1] + P[job, m]
        
        # For subsequent jobs:
        # C_curr[0] = C_prev[0] + P[job, 0]
        # C_curr[m] = max(C_curr[m-1], C_prev[m]) + P[job, m]
        
        # This is O(n_jobs * n_machines).
        
        if n_machines == 1:
            return float(np.sum(processing_times[seq, 0]))
            
        # We can try to vectorize the machine dimension if n_machines is small, 
        # but the max dependency prevents simple vectorization across m.
        
        # Let's just write a tight loop.
        C_prev = np.zeros(n_machines)
        
        for job in seq:
            p = processing_times[job]
            # Machine 0
            C_curr_0 = C_prev[0] + p[0]
            
            if n_machines == 1:
                C_prev[0] = C_curr_0
                continue
                
            # Other machines
            # We need to compute C_curr[m] sequentially
            # C_curr[m] = max(C_curr[m-1], C_prev[m]) + p[m]
            
            # We can store C_curr in a temporary array or just keep the last value
            # and update C_prev in place? No, we need C_prev[m] for the max.
            
            # Let's use a small array for C_curr
            # Actually, we can just update C_prev in place if we are careful?
            # C_curr[m] depends on C_curr[m-1] and C_prev[m].
            # If we update C_prev[m] to C_curr[m], we lose the old C_prev[m] for subsequent steps?
            # No, for machine m+1, we need C_curr[m] (which is the new C_prev[m]) and C_prev[m+1] (old).
            # So we can update in place if we go from m=1 to m-1?
            # Let's check:
            # C_curr[1] = max(C_curr[0], C_prev[1]) + p[1]
            # If we set C_prev[1] = C_curr[1], then for m=2:
            # C_curr[2] = max(C_curr[1], C_prev[2]) + p[2]
            # Here C_curr[1] is the new C_prev[1]. C_prev[2] is still the old one.
            # This works!
            
            C_prev[0] = C_curr_0
            curr_m1 = C_curr_0
            for m in range(1, n_machines):
                # max(curr_m1, C_prev[m]) + p[m]
                curr_m = max(curr_m1, C_prev[m]) + p[m]
                C_prev[m] = curr_m
                curr_m1 = curr_m
                
        return C_prev[-1]

    def neh_heuristic():
        """NEH (Nawaz, Enscore, Ham) heuristic."""
        totals = processing_times.sum(axis=1)
        order = list(np.argsort(-totals))
        
        seq = [order[0]]
        for job in order[1:]:
            best_pos = 0
            best_ms = float('inf')
            # NEH insertion: try all positions
            # Optimization: Use a simple loop
            for pos in range(len(seq) + 1):
                # Create trial sequence
                # To avoid list concatenation overhead, we can compute makespan incrementally?
                # For now, simple insertion and makespan calc.
                trial = seq[:pos] + [job] + seq[pos:]
                ms = makespan(trial)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            seq = seq[:best_pos] + [job] + seq[best_pos:]
        
        return seq

    def lpt_heuristic():
        """LPT: Longest Processing Time (sum of processing times)."""
        totals = processing_times.sum(axis=1)
        return list(np.argsort(-totals))

    def palmer_heuristic():
        """Palmer's priority rule: sort by (n - i - 0.5) * p_i - sum(p_j for j < i)"""
        n = n_jobs
        totals = processing_times.sum(axis=1)
        # Palmer's key for job j at position i (0-indexed):
        # Key = (n - i - 0.5) * p_j - sum_{k=0}^{i-1} p_k
        # This is tricky to sort directly because the position i changes.
        # Standard approximation: Sort by (n - 0.5) * p_j - sum(p_k for k != j)? No.
        # Let's use the standard Palmer index: 
        # For job j, compute sum of processing times.
        # Actually, a common implementation is:
        # Sort jobs by decreasing (n - 0.5) * p_j - sum_{k} p_k? No.
        # Let's use a simpler variant or skip if complex.
        # Alternative: Sort by difference of first and last machine?
        # Let's stick to LPT and SPT for diversity.
        return lpt_heuristic()

    def spt_heuristic():
        """SPT: Shortest Processing Time."""
        totals = processing_times.sum(axis=1)
        return list(np.argsort(totals))

    def local_search_2opt(seq, ms):
        """
        2-opt local search: Reverse segments to improve makespan.
        """
        n = len(seq)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Randomize the order of checking to escape local optima faster?
            # Or systematic? Systematic is better for convergence, random for diversity.
            # Let's do a systematic pass, but stop if no improvement in a full pass.
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Reverse segment [i, j]
                    new_seq = seq[:i] + seq[i:j+1][::-1] + seq[j+1:]
                    new_ms = makespan(new_seq)
                    if new_ms < ms:
                        seq = new_seq
                        ms = new_ms
                        improved = True
                        # Restart the inner loop or continue? 
                        # Continuing is okay, but the sequence changed.
                        # To be safe and simple, we can break and restart the outer loop?
                        # That's expensive. Let's just continue and hope for the best.
                        # Actually, modifying seq while iterating indices is dangerous if we rely on original indices.
                        # But we are checking all pairs. If we accept a move, the "best" neighbor might be different.
                        # Standard 2-opt: find the best move, apply it, restart.
                        # Or: find any improving move, apply it, restart.
                        # Let's do: find best improving move in this pass, apply it, restart pass.
                        # This is safer.
                        break
                if improved:
                    break
        return seq, ms

    # Multi-Start ILS
    best_seq = None
    best_ms = float('inf')
    
    # Initial heuristics
    heuristics = [neh_heuristic, lpt_heuristic, spt_heuristic]
    
    # Run heuristics
    for h in heuristics:
        if time.time() > deadline:
            break
        seq = h()
        ms = makespan(seq)
        if ms < best_ms:
            best_ms = ms
            best_seq = list(seq)
        
        # Local search on this start
        seq_ls, ms_ls = local_search_2opt(seq, ms)
        if ms_ls < best_ms:
            best_ms = ms_ls
            best_seq = list(seq_ls)

    # Iterated Local Search: Perturb best solution, local search, keep if better
    current_seq = list(best_seq)
    current_ms = best_ms
    
    # Parameters for ILS
    # Perturbation: Random k-swap or segment reversal
    # Let's use segment reversal of random length
    
    while time.time() < deadline:
        # Perturb
        # Choose a random segment to reverse
        length = random.randint(2, min(n_jobs, 10)) # Short reversals
        i = random.randint(0, n_jobs - length)
        j = i + length - 1
        
        perturbed_seq = current_seq[:i] + current_seq[i:j+1][::-1] + current_seq[j+1:]
        
        # Local Search
        ls_seq, ls_ms = local_search_2opt(perturbed_seq, makespan(perturbed_seq))
        
        # Accept if better than global best
        if ls_ms < best_ms:
            best_ms = ls_ms
            best_seq = list(ls_seq)
            current_seq = list(ls_seq)
            current_ms = ls_ms
        # Optionally accept worse solutions with probability (SA-like) to escape, 
        # but ILS usually just keeps the best or uses a fixed acceptance.
        # Let's keep the best for simplicity and stability.
        
        # Check time
        if time.time() > deadline:
            break

    return list(best_seq)
# EVOLVE-BLOCK-END
