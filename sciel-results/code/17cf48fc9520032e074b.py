"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact QAP cost for assignment p."""
    D = b[np.ix_(p, p)]
    return int(np.sum(a * D))

def _swap_cost_delta(a, b, p, i, j):
    """
    Compute the cost change for swapping positions i and j in permutation p.
    Uses the formula:
    delta = sum_{k != i,j} a[i,k]*(b[p[j],p[k]] - b[p[i],p[k]]) 
          + sum_{k != i,j} a[k,i]*(b[p[j],p[k]] - b[p[i],p[k]])
          + a[i,i]*(b[p[j],p[j]] - b[p[i],p[i]])
          + a[j,j]*(b[p[i],p[i]] - b[p[j],p[j]])
          + a[i,j]*(b[p[j],p[i]] - b[p[i],p[j]])
          + a[j,i]*(b[p[i],p[j]] - b[p[j],p[i]])
    
    Let A = a, B = b.
    Let pi = p[i], pj = p[j].
    
    The change in cost is:
    C_new - C_old = 
      sum_{k != i,j} a[i,k] * (B[pj, p[k]] - B[pi, p[k]])
    + sum_{k != i,j} a[k,i] * (B[pj, p[k]] - B[pi, p[k]])
    + a[i,i] * (B[pj, pj] - B[pi, pi])
    + a[j,j] * (B[pi, pi] - B[pj, pj])
    + a[i,j] * (B[pj, pi] - B[pi, pj])
    + a[j,i] * (B[pi, pj] - B[pj, pi])
    
    We can rewrite the sums using vectorized operations.
    Let diff_B = B[pj, :] - B[pi, :]  (shape n)
    Let row_i = a[i, :]  (shape n)
    Let col_i = a[:, i]  (shape n)
    
    Sum1 = sum_{k != i,j} row_i[k] * diff_B[k]
    Sum2 = sum_{k != i,j} col_i[k] * diff_B[k]
    
    Sum1 + Sum2 = sum_k (row_i[k] + col_i[k]) * diff_B[k] - (row_i[i] + col_i[i])*diff_B[i] - (row_i[j] + col_i[j])*diff_B[j]
    
    Note: diff_B[i] = B[pj, pi] - B[pi, pi]
          diff_B[j] = B[pj, pj] - B[pi, pj]
    
    The diagonal and cross terms in the delta formula are:
    T_diag = a[i,i]*(B[pj,pj] - B[pi,pi]) + a[j,j]*(B[pi,pi] - B[pj,pj])
           = (a[i,i] - a[j,j]) * (B[pj,pj] - B[pi,pi])
    T_cross = a[i,j]*(B[pj,pi] - B[pi,pj]) + a[j,i]*(B[pi,pj] - B[pj,pi])
            = (a[i,j] - a[j,i]) * (B[pj,pi] - B[pi,pj])
    
    So, delta = (Sum1 + Sum2) + T_diag + T_cross
    
    Let's compute Sum1 + Sum2 carefully.
    Let V = a[i, :] + a[:, i]  (shape n)
    Let W = B[pj, :] - B[pi, :]  (shape n)
    
    Sum_all = np.dot(V, W)
    
    Correction for k=i:
    term_i = V[i] * W[i]
    Correction for k=j:
    term_j = V[j] * W[j]
    
    Sum1_plus_Sum2 = Sum_all - term_i - term_j
    
    Then delta = Sum1_plus_Sum2 + T_diag + T_cross
    
    This is O(n) per swap.
    """
    pi = p[i]
    pj = p[j]
    
    # Vector for A contribution
    V = a[i, :] + a[:, i]
    # Vector for B difference
    W = b[pj, :] - b[pi, :]
    
    Sum_all = np.dot(V, W)
    
    term_i = V[i] * W[i]
    term_j = V[j] * W[j]
    
    Sum1_plus_Sum2 = Sum_all - term_i - term_j
    
    T_diag = (a[i, i] - a[j, j]) * (b[pj, pj] - b[pi, pi])
    T_cross = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    delta = Sum1_plus_Sum2 + T_diag + T_cross
    
    return int(delta)

def _local_search(a, b, p, time_limit, start_time):
    """
    Optimized 2-opt local search for QAP.
    Uses first-improvement strategy with O(n) delta evaluation.
    """
    n = len(p)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    
    # Precompute nothing heavy, just use the fast delta function.
    # To speed up, we can iterate over pairs.
    # For n=256, there are ~32k pairs. Each delta is O(n)=256 ops.
    # 32k * 256 = 8M ops per pass.
    # In Python, this might be slow. 8M numpy ops per pass... 
    # np.dot is fast, but the overhead of the function call and indexing might add up.
    # Let's try to vectorize the delta calculation for a fixed i against all j > i.
    
    # For a fixed i, we want to find j > i that minimizes delta.
    # delta(i, j) = ...
    # Let's precompute B_diffs? No, B_diffs depend on j (pj).
    
    # Alternative: Use the vectorized approach from the inspiration but fix the logic.
    # The inspiration's vectorized delta was:
    # S = sum((A_sym[i] - A_sym[js]) * (D[js] - D[i]), axis=1)
    # ... corrections ...
    # This is O(n^2) per i, which is O(n^3) total per pass. Too slow.
    
    # The O(n) per swap is better if we can do it fast.
    # Let's stick to the O(n) per swap but try to optimize the inner loop.
    # We will iterate i, then j.
    
    # To make it faster, we can precompute a_sym? No, V = a[i,:] + a[:,i] changes with i.
    # But a[:, i] is just a column.
    
    # Let's try a simple loop first and see if it's fast enough.
    # If not, we will need a more clever vectorization.
    
    # Actually, let's use the vectorized delta for a FIXED i, but compute it for ALL j > i at once.
    # Can we?
    # delta(i, j) = dot(V_i, W_j) - V_i[i]*W_j[i] - V_i[j]*W_j[j] + T_diag(i,j) + T_cross(i,j)
    # Where V_i = a[i,:] + a[:,i]
    # W_j = b[p[j],:] - b[p[i],:]
    
    # W_j = b[p[j],:] - b[pi,:]
    # Let B_pi = b[pi, :]
    # Let B_js = b[p[js], :]  (shape (n-i-1, n))
    # W_js = B_js - B_pi  (shape (n-i-1, n))
    
    # dot(V_i, W_j) for all j:
    # This is V_i @ W_js.T  (shape (n-i-1,))
    
    # Term_i = V_i[i] * W_j[i]
    # W_j[i] = b[p[j], pi] - b[pi, pi]
    # Let B_pi_pi = b[pi, pi]
    # Let B_js_pi = b[p[js], pi]  (shape (n-i-1,))
    # W_js_i = B_js_pi - B_pi_pi
    # Term_i_vec = V_i[i] * W_js_i  (shape (n-i-1,))
    
    # Term_j = V_i[j] * W_j[j]
    # V_i[j] = a[i, j] + a[j, i]
    # W_j[j] = b[p[j], p[j]] - b[pi, p[j]]
    # This is tricky because V_i[j] depends on j.
    # V_i is fixed for fixed i.
    # V_i[j] is the j-th element of V_i.
    # So Term_j_vec = V_i[js] * (b[p[js], p[js]] - b[pi, p[js]])
    
    # T_diag(i,j) = (a[i,i] - a[j,j]) * (b[pj,pj] - b[pi,pi])
    # T_cross(i,j) = (a[i,j] - a[j,i]) * (b[pj,pi] - b[pi,pj])
    
    # This is getting complicated to vectorize fully for all j at once because of the j-dependent terms in V_i[j] and a[j,j], etc.
    # However, we can precompute some parts.
    
    # Let's try a hybrid: Vectorize the main dot product, then correct with scalar/vector ops.
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                return p, current_cost
            
            pi = p[i]
            js = np.arange(i + 1, n)
            pjs = p[js]
            
            # V_i = a[i, :] + a[:, i]
            V_i = a[i, :] + a[:, i]
            
            # W_js = b[pjs, :] - b[pi, :]
            W_js = b[pjs, :] - b[pi, :]  # (m, n) where m = n-i-1
            
            # Main term: V_i @ W_js.T
            main_terms = W_js @ V_i  # (m,)
            
            # Correction term_i: V_i[i] * (b[pjs, pi] - b[pi, pi])
            term_i = V_i[i] * (b[pjs, pi] - b[pi, pi])  # (m,)
            
            # Correction term_j: V_i[js] * (b[pjs, pjs] - b[pi, pjs])
            # V_i[js] is V_i[js]
            # b[pjs, pjs] is diag(b[np.ix_(pjs, pjs)])
            # b[pi, pjs] is b[pi, pjs]
            term_j = V_i[js] * (np.diag(b[np.ix_(pjs, pjs)]) - b[pi, pjs])  # (m,)
            
            # T_diag: (a[i,i] - a[js, js]) * (b[pjs, pjs] - b[pi, pi])
            t_diag = (a[i, i] - np.diag(a[np.ix_(js, js)])) * (np.diag(b[np.ix_(pjs, pjs)]) - b[pi, pi])  # (m,)
            
            # T_cross: (a[i, js] - a[js, i]) * (b[pjs, pi] - b[pi, pjs])
            t_cross = (a[i, js] - a[js, i]) * (b[pjs, pi] - b[pi, pjs])  # (m,)
            
            deltas = main_terms - term_i - term_j + t_diag + t_cross
            
            min_idx = np.argmin(deltas)
            min_delta = deltas[min_idx]
            
            if min_delta < 0:
                j = js[min_idx]
                p[i], p[j] = p[j], p[i]
                current_cost += int(min_delta)
                improved = True
        
        if not improved:
            break
    
    return p, current_cost

def _hungarian_init(a, b):
    """
    Generate initial assignment using Hungarian algorithm on a proxy cost matrix.
    Proxy cost: C[i, j] = - (flow_centrality[i] * distance_centrality[j])
    flow_centrality[i] = sum(a[i, :])
    distance_centrality[j] = sum(b[:, j])
    """
    n = len(a)
    flow_cent = a.sum(axis=1)
    dist_cent = b.sum(axis=1)
    
    # We want to maximize sum(flow_cent[i] * dist_cent[p[i]])
    # So minimize sum(-flow_cent[i] * dist_cent[p[i]])
    # Cost matrix M[i, j] = -flow_cent[i] * dist_cent[j]
    M = np.outer(-flow_cent, dist_cent)
    
    row_ind, col_ind = linear_sum_assignment(M)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p

def _perturb(p, k):
    """Randomly swap k pairs of positions."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = random.sample(range(n), 2)
        p[i], p[j] = p[j], p[i]
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 145  # Leave 5s buffer for the 150s limit
    
    best_p = None
    best_cost = np.inf
    
    # Phase 1: Hungarian Initialization + Local Search
    # This provides a structurally distinct starting point.
    p_init = _hungarian_init(a, b)
    p_ls, cost_ls = _local_search(a, b, p_init, time_budget * 0.2, start_time)
    
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls.copy()
    
    # Phase 2: Multi-start Random Initializations
    # Use remaining time for random restarts.
    elapsed = time.time() - start_time
    remaining_time = time_budget - elapsed
    
    # Allocate 30% of remaining time to a few random restarts
    restart_time = remaining_time * 0.3
    num_restarts = 5
    time_per_restart = restart_time / num_restarts
    
    for _ in range(num_restarts):
        if time.time() - start_time > start_time + restart_time:
            break
        p_rand = np.random.permutation(n)
        p_ls, cost_ls = _local_search(a, b, p_rand, time.time() + time_per_restart, start_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Phase 3: Iterated Local Search (ILS)
    # Use the remaining time for ILS on the best solution found.
    elapsed = time.time() - start_time
    remaining_time = time_budget - elapsed
    
    if remaining_time < 10:
        return best_p.tolist()
    
    consecutive_no_improve = 0
    base_perturb = 3
    
    while time.time() - start_time < time_budget - 2:
        # Check time
        if time.time() - start_time > time_budget - 2:
            break
        
        # Adaptive perturbation
        if consecutive_no_improve > 10:
            perturb_k = min(15, base_perturb + consecutive_no_improve // 2)
        else:
            perturb_k = base_perturb
            
        p_pert = _perturb(best_p, perturb_k)
        
        # Allocate time for this LS run
        # Use a chunk of the remaining time, ensuring we can finish.
        # Let's say we give it up to 5 seconds or 10% of remaining time, whichever is smaller but at least 1s.
        ls_time = min(5.0, remaining_time * 0.1)
        if ls_time < 1.0:
            ls_time = 1.0
            
        end_time_ls = time.time() + ls_time
        
        p_ls, cost_ls = _local_search(a, b, p_pert, end_time_ls, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
        
        # Update remaining time
        remaining_time = time_budget - (time.time() - start_time)
        
        if remaining_time < 5:
            break
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
