import numpy as np
from scipy.optimize import differential_evolution
import time

def _calc_makespan(perm, processing_times):
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return 0
    C_prev = np.zeros(n_machines)
    for job_idx in perm:
        C_curr = np.zeros(n_machines)
        for m in range(n_machines):
            start = max(C_prev[m], C_curr[m-1] if m > 0 else 0)
            C_curr[m] = start + processing_times[job_idx, m]
        C_prev = C_curr
    return C_prev[-1]

def _constructive_heuristic(processing_times):
    """
    Generate a good initial permutation using a greedy heuristic.
    Uses a combination of total processing time and critical path approximation.
    """
    n_jobs, n_machines = processing_times.shape
    # Simple heuristic: Sort by sum of processing times (descending)
    # Or use Johnson's rule for 2 machines, generalize for others.
    # Here we use a simple "Critical Ratio" like approach or just Sum.
    sums = np.sum(processing_times, axis=1)
    # Heuristic: Jobs with larger sums tend to be critical.
    # Let's try sorting by sum descending.
    perm = np.argsort(-sums, kind='stable')
    return list(perm)

def _local_search_insertion(perm, processing_times, time_limit=10.0):
    """
    Insertion local search with time limit.
    """
    n_jobs = len(perm)
    best_perm = list(perm)
    best_makespan = _calc_makespan(best_perm, processing_times)
    start_time = time.time()
    improved = True
    
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        for i in range(n_jobs):
            if time.time() - start_time > time_limit:
                break
            for j in range(n_jobs):
                if i == j:
                    continue
                new_perm = list(best_perm)
                job = new_perm.pop(i)
                insert_idx = j - 1 if j > i else j
                new_perm.insert(insert_idx, job)
                
                if new_perm == best_perm:
                    continue
                    
                new_makespan = _calc_makespan(new_perm, processing_times)
                if new_makespan < best_makespan:
                    best_perm = new_perm
                    best_makespan = new_makespan
                    improved = True
                    break
            if improved:
                break
                
    return best_perm, best_makespan

def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # 1. Generate initial permutations
    perm_heuristic = _constructive_heuristic(processing_times)
    makespan_heuristic = _calc_makespan(perm_heuristic, processing_times)
    
    best_perm = perm_heuristic
    best_makespan = makespan_heuristic
    
    # 2. Use DE to find a good weight vector
    # We only run DE if n_jobs is small enough to be fast, or if we have time
    # DE on N dimensions is expensive. Let's cap the effort.
    if n_jobs <= 30:
        def objective(weights):
            perm = np.argsort(-weights, kind='stable')
            return _calc_makespan(perm, processing_times)
            
        bounds = [(0, 1)] * n_jobs
        
        try:
            result = differential_evolution(
                objective, 
                bounds, 
                popsize=10, 
                maxiter=30, 
                tol=1e-4,
                seed=42,
                init='sobol'
            )
            best_weights = result.x
            perm_de = np.argsort(-best_weights, kind='stable')
            makespan_de = _calc_makespan(perm_de, processing_times)
            
            if makespan_de < best_makespan:
                best_perm = list(perm_de)
                best_makespan = makespan_de
        except Exception:
            pass # Fallback to heuristic
    
    # 3. Local Search to polish
    # Use a time limit to ensure we don't exceed total 120s
    # Assume DE took some time, leave 10s for local search
    final_perm, _ = _local_search_insertion(best_perm, processing_times, time_limit=10.0)
    
    return list(final_perm)
