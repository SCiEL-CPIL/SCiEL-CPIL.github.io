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
            
            # Early termination: if current completion time already exceeds upper bound,
            # the final makespan will definitely be > upper_bound.
            if C_curr[m] > upper_bound:
                return C_curr[m]
        C_prev = C_curr
    return float(C_prev[-1])

def _decode_weights(weights, n_jobs):
    """
    Decode continuous weights into a permutation.
    Higher weight = earlier position (or vice versa, doesn't matter as long as consistent).
    We use argsort.
    """
    # Ties are broken by index, which is fine.
    return np.argsort(-weights, kind='stable')

def _local_search(perm, processing_times, time_limit=10.0):
    """
    Enhanced Local Search with Insertion and 2-Swap moves.
    Uses Best-Improvement strategy within a time limit.
    Utilizes early termination in makespan calculation.
    """
    n_jobs = len(perm)
    best_perm = list(perm)
    best_makespan = _calc_makespan(best_perm, processing_times)
    start_time = time.time()
    
    while True:
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        best_new_perm = None
        best_new_makespan = best_makespan
        
        # 1. Try Insertion moves
        for i in range(n_jobs):
            if time.time() - start_time > time_limit:
                break
            job = best_perm[i]
            for j in range(n_jobs):
                if i == j:
                    continue
                # Create new permutation
                new_perm = best_perm.copy()
                new_perm.pop(i)
                # Adjust insertion index if removing from before
                insert_idx = j - 1 if j > i else j
                new_perm.insert(insert_idx, job)
                
                new_makespan = _calc_makespan(new_perm, processing_times, upper_bound=best_new_makespan)
                
                if new_makespan < best_new_makespan:
                    best_new_makespan = new_makespan
                    best_new_perm = new_perm
        
        # 2. Try 2-Swap moves (if time remains)
        if time.time() - start_time < time_limit - 0.5:
            for i in range(n_jobs):
                if time.time() - start_time > time_limit:
                    break
                for j in range(i + 1, n_jobs):
                    new_perm = best_perm.copy()
                    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                    new_makespan = _calc_makespan(new_perm, processing_times, upper_bound=best_new_makespan)
                    if new_makespan < best_new_makespan:
                        best_new_makespan = new_makespan
                        best_new_perm = new_perm

        if best_new_perm is not None and best_new_makespan < best_makespan:
            best_perm = best_new_perm
            best_makespan = best_new_makespan
            improved = True
        else:
            break
            
    return best_perm, best_makespan

def _cds_heuristic(processing_times):
    n_jobs, n_machines = processing_times.shape
    sums = np.sum(processing_times, axis=1)
    sorted_indices = np.argsort(-sums, kind='stable')
    
    best_perm = None
    best_ms = float('inf')
    
    for k in range(1, n_jobs + 1):
        prefix = list(sorted_indices[:k])
        # Remaining jobs in original order
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
    
    # 1. Initial Heuristics: CDS, Sum, Random
    starts = []
    
    # Heuristic 1: CDS
    perm_cds, ms_cds = _cds_heuristic(processing_times)
    starts.append((perm_cds, ms_cds))
    
    # Heuristic 2: Sum descending
    sums = np.sum(processing_times, axis=1)
    perm_sum = np.argsort(-sums, kind='stable')
    ms_sum = _calc_makespan(perm_sum, processing_times)
    starts.append((list(perm_sum), ms_sum))
    
    # Heuristic 3: Random
    np.random.seed(42)
    perm_rand = np.random.permutation(n_jobs)
    ms_rand = _calc_makespan(perm_rand, processing_times)
    starts.append((list(perm_rand), ms_rand))
    
    best_perm = None
    best_makespan = float('inf')
    
    for p, ms in starts:
        if ms < best_makespan:
            best_makespan = ms
            best_perm = p
            
    # 2. Optimize Weights using DE
    # We map weights to permutations.
    # We run DE to find weights that yield a good permutation.
    
    current_best_de = {'val': best_makespan}
    
    def objective(weights):
        perm = _decode_weights(weights, n_jobs)
        ms = _calc_makespan(perm, processing_times, upper_bound=current_best_de['val'])
        if ms < current_best_de['val']:
            current_best_de['val'] = ms
        return ms
        
    bounds = [(0, 1)] * n_jobs
    
    # Adjust DE parameters based on problem size
    if n_jobs <= 20:
        maxiter = 50
        popsize = 15
    elif n_jobs <= 50:
        maxiter = 30
        popsize = 10
    else:
        maxiter = 20
        popsize = 8
        
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            popsize=popsize, 
            maxiter=maxiter, 
            tol=1e-4,
            seed=42,
            init='sobol',
            polish=False
        )
        
        best_weights = result.x
        perm_de = _decode_weights(best_weights, n_jobs)
        makespan_de = _calc_makespan(perm_de, processing_times)
        
        if makespan_de < best_makespan:
            best_perm = list(perm_de)
            best_makespan = makespan_de
            
    except Exception:
        pass # Fallback to initial heuristic
    
    # 3. Local Search to polish
    # Allocate remaining time to LS
    final_perm, _ = _local_search(best_perm, processing_times, time_limit=30.0)
    
    return list(final_perm)
