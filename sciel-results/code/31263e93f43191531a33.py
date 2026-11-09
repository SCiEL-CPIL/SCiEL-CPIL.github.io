import numpy as np
from scipy.optimize import differential_evolution
import time

def _calc_makespan(perm, processing_times, upper_bound=float('inf')):
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return 0.0
    
    C_prev = np.zeros(n_machines)
    for job_idx in perm:
        p = processing_times[job_idx]
        C_curr = np.empty(n_machines)
        for m in range(n_machines):
            if m == 0:
                C_curr[0] = C_prev[0] + p[0]
            else:
                C_curr[m] = max(C_prev[m], C_curr[m-1]) + p[m]
            
            if C_curr[m] > upper_bound:
                return C_curr[m]
        C_prev = C_curr
    return float(C_prev[-1])

def _decode_weights(weights, n_jobs):
    return np.argsort(-weights, kind='stable')

def _local_search_first_improvement(perm, processing_times, time_limit=5.0):
    """
    Fast Local Search using First-Improvement strategy for Insertion moves.
    """
    n_jobs = len(perm)
    best_perm = list(perm)
    best_makespan = _calc_makespan(best_perm, processing_times)
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        # Try all insertion moves
        for i in range(n_jobs):
            if time.time() - start_time > time_limit:
                break
            job = best_perm[i]
            # Remove job
            new_perm = best_perm.copy()
            new_perm.pop(i)
            
            # Try inserting at every position
            for j in range(n_jobs):
                if i == j:
                    continue
                # Adjust index because we popped from i
                # If j > i, the index shifts left by 1 after pop
                # If j < i, index remains same
                insert_idx = j - 1 if j > i else j
                
                # Skip if inserting back at same position
                if insert_idx == i:
                    continue
                
                # Create candidate
                cand_perm = new_perm.copy()
                cand_perm.insert(insert_idx, job)
                
                # Calculate makespan
                cand_ms = _calc_makespan(cand_perm, processing_times, upper_bound=best_makespan)
                
                if cand_ms < best_makespan:
                    best_perm = cand_perm
                    best_makespan = cand_ms
                    improved = True
                    break # First improvement found, restart scan
    
    return best_perm, best_makespan

def _cds_heuristic(processing_times):
    n_jobs, n_machines = processing_times.shape
    sums = np.sum(processing_times, axis=1)
    sorted_indices = np.argsort(-sums, kind='stable')
    
    best_perm = None
    best_ms = float('inf')
    
    for k in range(1, n_jobs + 1):
        prefix = list(sorted_indices[:k])
        remaining = [j for j in range(n_jobs) if j not in set(prefix)]
        candidate = prefix + remaining
        
        ms = _calc_makespan(candidate, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_perm = candidate
    
    if best_perm is None:
        best_perm = list(range(n_jobs))
    return best_perm, best_ms

def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # 1. Generate Initial Heuristics to get good starting points
    # We will use these to initialize the DE population or as fallbacks
    
    # Heuristic 1: CDS
    perm_cds, ms_cds = _cds_heuristic(processing_times)
    
    # Heuristic 2: SPT (Shortest Processing Time)
    sums_spt = np.sum(processing_times, axis=1)
    perm_spt = np.argsort(sums_spt, kind='stable')
    ms_spt = _calc_makespan(perm_spt, processing_times)
    
    # Heuristic 3: LPT (Longest Processing Time)
    perm_lpt = np.argsort(-sums_spt, kind='stable')
    ms_lpt = _calc_makespan(perm_lpt, processing_times)
    
    # Heuristic 4: Random
    np.random.seed(42)
    perm_rand = np.random.permutation(n_jobs)
    ms_rand = _calc_makespan(perm_rand, processing_times)
    
    starts = [
        (perm_cds, ms_cds),
        (list(perm_spt), ms_spt),
        (list(perm_lpt), ms_lpt),
        (list(perm_rand), ms_rand)
    ]
    
    best_perm = None
    best_makespan = float('inf')
    
    for p, ms in starts:
        if ms < best_makespan:
            best_makespan = ms
            best_perm = p
    
    # 2. Optimize Weights using Differential Evolution
    # The weights determine the priority list.
    
    current_best_de = {'val': best_makespan}
    
    def objective(weights):
        perm = _decode_weights(weights, n_jobs)
        ms = _calc_makespan(perm, processing_times, upper_bound=current_best_de['val'])
        if ms < current_best_de['val']:
            current_best_de['val'] = ms
        return ms
        
    bounds = [(0, 1)] * n_jobs
    
    # Adjust DE parameters
    if n_jobs <= 20:
        maxiter = 100
        popsize = 20
    elif n_jobs <= 50:
        maxiter = 50
        popsize = 15
    else:
        maxiter = 30
        popsize = 10
        
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            popsize=popsize, 
            maxiter=maxiter, 
            tol=1e-5,
            seed=42,
            init='sobol',
            polish=False,
            mutation=(0.5, 1.0),
            recombination=0.7
        )
        
        best_weights = result.x
        perm_de = _decode_weights(best_weights, n_jobs)
        makespan_de = _calc_makespan(perm_de, processing_times)
        
        if makespan_de < best_makespan:
            best_perm = list(perm_de)
            best_makespan = makespan_de
            
    except Exception:
        pass
    
    # 3. Local Search to polish
    # Use the fast first-improvement LS
    final_perm, _ = _local_search_first_improvement(best_perm, processing_times, time_limit=30.0)
    
    return list(final_perm)
