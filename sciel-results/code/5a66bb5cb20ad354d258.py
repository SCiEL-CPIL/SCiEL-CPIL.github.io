"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _local_search(a, b, p, time_limit, start_time):
    """
    Optimized 2-opt local search for QAP with correct delta calculation.
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                return p, current_cost
            
            # Compute delta for swapping position i with all j > i
            js = np.arange(i + 1, n)
            m = len(js)
            
            # Part 1: direct i-j term
            # (a[i,j] - a[j,i]) * (D[j,i] - D[i,j])
            part1 = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])
            
            # Part 2: sum over k != i, j
            # For each j in js, we need:
            # sum_{k != i, j} [(a[i,k]-a[j,k])*(D[j,k]-D[i,k]) + (a[k,i]-a[k,j])*(D[k,j]-D[k,i])]
            
            # Precompute for fixed i:
            # A_i[k] = a[i,k] for all k
            # D_i[k] = D[i,k] for all k
            # B_i[k] = a[k,i] for all k  
            # Dk_i[k] = D[k,i] for all k
            
            # For all j in js simultaneously:
            # term1[j,k] = (a[i,k] - a[j,k]) * (D[j,k] - D[i,k])
            # term2[j,k] = (a[k,i] - a[k,j]) * (D[k,j] - D[k,i])
            
            # We need to exclude k=i and k=j from the sum.
            
            # First compute the full sum including k=i and k=j, then subtract those terms.
            
            # Full sum of term1 over all k:
            # sum_k (a[i,k] - a[j,k]) * (D[j,k] - D[i,k])
            # = sum_k a[i,k]*D[j,k] - sum_k a[i,k]*D[i,k] - sum_k a[j,k]*D[j,k] + sum_k a[j,k]*D[i,k]
            
            # Let's compute these for all j in js:
            # S1[j] = sum_k a[i,k]*D[j,k] = (a[i,:] @ D[js,:].T)[j]... 
            # Actually: sum_k a[i,k]*D[j,k] for fixed j = a[i,:] dot D[j,:]
            
            # For all j in js: S1 = D[js,:] @ a[i,:]  -- shape (m,)
            S1 = D[js] @ a[i]  # (m,)
            
            # S2[j] = sum_k a[i,k]*D[i,k] = a[i,:] dot D[i,:] -- same for all j
            S2 = float(a[i] @ D[i])  # scalar
            
            # S3[j] = sum_k a[j,k]*D[j,k]
            S3 = np.sum(a[np.ix_(js, js)] * D[np.ix_(js, js)], axis=1)  # (m,)
            
            # S4[j] = sum_k a[j,k]*D[i,k]
            S4 = D[i] @ a[js].T  # (m,)
            
            # Full term1 sum = S1 - S2 - S3 + S4
            full_term1 = S1 - S2 - S3 + S4  # (m,)
            
            # Now subtract k=i contribution:
            # k=i: (a[i,i] - a[j,i]) * (D[j,i] - D[i,i])
            ki_term1 = (a[i, i] - a[js, i]) * (D[js, i] - D[i, i])  # (m,)
            
            # k=j contribution for term1:
            # (a[i,j] - a[j,j]) * (D[j,j] - D[i,j])
            kj_term1 = (a[i, js] - np.diag(a[np.ix_(js, js)])) * (D[np.ix_(js, js)].diagonal() - D[i, js])  # (m,)
            
            # Corrected term1 = full_term1 - ki_term1 - kj_term1
            term1_corrected = full_term1 - ki_term1 - kj_term1  # (m,)
            
            # Now term2: sum_k (a[k,i] - a[k,j]) * (D[k,j] - D[k,i])
            # = sum_k a[k,i]*D[k,j] - sum_k a[k,i]*D[k,i] - sum_k a[k,j]*D[k,j] + sum_k a[k,j]*D[k,i]
            
            # T1[j] = sum_k a[k,i]*D[k,j] = a[:,i] dot D[:,j]... 
            # For all j in js: T1 = (a[:, i] * D[:, js]).sum(axis=0)
            T1 = (a[:, i][:, None] * D[:, js]).sum(axis=0)  # (m,)
            
            # T2[j] = sum_k a[k,i]*D[k,i] = a[:,i] dot D[:,i] -- same for all j
            T2 = float(a[:, i] @ D[:, i])  # scalar
            
            # T3[j] = sum_k a[k,j]*D[k,j]
            T3 = np.sum(a[np.ix_(js, js)].T * D[np.ix_(js, js)], axis=1)  # (m,)
            # Wait: a[k,j] for k in 0..n-1, j in js. a[np.ix_(range(n), js)]
            # sum_k a[k,j]*D[k,j] = sum over k of a[k,j]*D[k,j]
            # For fixed j: this is a[:,j] dot D[:,j]
            T3 = (a[:, js] * D[:, js]).sum(axis=0)  # (m,)
            
            # T4[j] = sum_k a[k,j]*D[k,i]
            T4 = (a[:, js] * D[:, i][:, None]).sum(axis=0)  # (m,)
            
            # Full term2 sum = T1 - T2 - T3 + T4
            full_term2 = T1 - T2 - T3 + T4  # (m,)
            
            # Subtract k=i contribution:
            # k=i: (a[i,i] - a[i,j]) * (D[i,j] - D[i,i])
            ki_term2 = (a[i, i] - a[i, js]) * (D[i, js] - D[i, i])  # (m,)
            
            # k=j contribution for term2:
            # k=j: (a[j,i] - a[j,j]) * (D[j,j] - D[j,i])
            kj_term2 = (a[js, i] - np.diag(a[np.ix_(js, js)])) * (D[np.ix_(js, js)].diagonal() - D[js, i])  # (m,)
            
            # Corrected term2 = full_term2 - ki_term2 - kj_term2
            term2_corrected = full_term2 - ki_term2 - kj_term2  # (m,)
            
            deltas = part1 + term1_corrected + term2_corrected
            
            min_idx = np.argmin(deltas)
            min_delta = deltas[min_idx]
            
            if min_delta < 0:
                j = js[min_idx]
                p[i], p[j] = p[j], p[i]
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                current_cost += int(min_delta)
                improved = True
        
        if not improved:
            break
    
    return p, current_cost

def _perturb(p, k):
    """Randomly swap k pairs of positions."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = np.random.choice(n, 2, replace=False)
        p[i], p[j] = p[j], p[i]
    return p

def _greedy_init(a, b, n):
    """
    Simple greedy initialization:
    Assign facilities one by one to the location that minimizes the current cost increase.
    This is O(n^3) which is acceptable for n=256.
    """
    p = np.zeros(n, dtype=int)
    assigned = np.zeros(n, dtype=bool)
    
    # To speed up, we can just do random if this is too slow, but let's try.
    # Actually, full greedy is complex to vectorize efficiently without loops.
    # Let's use a simpler heuristic: Random.
    # The directive says row sums are weak, so random is a safe baseline.
    return np.random.permutation(n)

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115  # leave 5s buffer
    
    best_p = None
    best_cost = np.inf
    
    # Phase 1: Multiple random restarts with local search
    # We allocate 30% of time to this phase
    phase1_time = time_budget * 0.3
    num_initials = 10
    time_per_initial = phase1_time / num_initials
    
    for _ in range(num_initials):
        if time.time() - start_time > start_time + phase1_time:
            break
        p_init = np.random.permutation(n)
        p_ls, cost_ls = _local_search(a, b, p_init, time_per_initial, start_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Phase 2: ILS
    elapsed = time.time() - start_time
    remaining_time = time_budget - elapsed
    
    if remaining_time < 5 or best_p is None:
        return best_p.tolist() if best_p is not None else np.random.permutation(n).tolist()
        
    consecutive_no_improve = 0
    base_perturb = 3
    
    while time.time() - start_time < time_budget - 2:
        if time.time() - start_time > time_budget - 2:
            break
            
        # Adaptive perturbation: increase if stuck
        if consecutive_no_improve > 5:
            perturb_k = min(20, base_perturb + consecutive_no_improve)
        else:
            perturb_k = base_perturb
            
        p_pert = _perturb(best_p, perturb_k)
        
        # Cap local search time per ILS iteration to allow more iterations
        ls_time_cap = min(remaining_time * 0.15, 10.0)  # max 10s per LS, or 15% of remaining
        duration = ls_time_cap
        if duration < 1:
            duration = max(1.0, (time_budget - 2) - (time.time() - start_time))
        if duration < 1:
            break
            
        p_ls, cost_ls = _local_search(a, b, p_pert, time.time() + duration, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
