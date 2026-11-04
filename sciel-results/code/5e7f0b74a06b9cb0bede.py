"""
Hybrid QAP solver for wil100: Riemannian CQAP initialization + vectorized steepest-descent
local search + simulated annealing polish.
Combines the structural idea of continuous relaxation on the Stiefel manifold for a
high-quality initial permutation with the fast exact local search and SA machinery.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _greedy_initial(a, b):
    """
    Create an initial permutation using a greedy heuristic.
    Matches facilities with high flow to locations with low distance.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Sort facilities by descending flow sum
    heavy = np.argsort(-a_row_sums, kind="stable")
    # Sort locations by ascending distance sum
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def _random_initial(n):
    """Generate a random permutation."""
    return np.random.permutation(n).astype(np.int64)

def _compute_swap_delta(a, b, p, i, j):
    """
    Compute the cost delta for swapping positions i and j in permutation p.
    Returns the integer delta.
    """
    pi = p[i]
    pj = p[j]
    
    # Term 1: sum_{k != i,j} (a[k,i] - a[k,j]) * (b[p[k],p[j]] - b[p[k],p[i]])
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_all_T1 = np.dot(diff_a_col, diff_b_row)
    
    # Subtract contributions from k=i and k=j
    term_1_i = (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    term_1_j = (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    T1 = sum_all_T1 - term_1_i - term_1_j
    
    # Term 2: sum_{k != i,j} (a[i,k] - a[j,k]) * (b[p[j],p[k]] - b[p[i],p[k]])
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_all_T2 = np.dot(diff_a_row, diff_b_col)
    
    # Subtract contributions from k=i and k=j
    term_2_i = (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    term_2_j = (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    T2 = sum_all_T2 - term_2_i - term_2_j
    
    # Term 3: (a[i,j] - a[j,i]) * (b[p[j],p[i]] - b[p[i],p[j]])
    T3 = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    return T1 + T2 + T3

def _compute_delta_matrix(a, b, p):
    """
    Compute the full n x n matrix of swap deltas.
    D[i, j] is the cost change if we swap positions i and j.
    Uses O(n^3) matrix multiplications, which is fast for n=100.
    """
    n = len(p)
    Bp = b[np.ix_(p, p)]
    
    # M1[i, j] = sum_k a[k, i] b[p[k], p[j]]
    M1 = a.T @ Bp
    
    # M2[i, j] = sum_k a[i, k] b[p[k], p[j]]
    M2 = a @ Bp
    
    # Term 1 matrix
    T1_raw = M1 - np.diag(M1)[:, None] - np.diag(M1)[None, :] + M1.T
    
    diag_a = np.diag(a)
    diag_Bp = np.diag(Bp)
    
    C1 = (diag_a[:, None] - a) * (Bp - diag_Bp[:, None])
    C2 = (a.T - diag_a[None, :]) * (diag_Bp[None, :] - Bp.T)
    
    T1 = T1_raw - C1 - C2
    
    # Term 2 matrix
    T2_raw = M2 - np.diag(M2)[:, None] - np.diag(M2)[None, :] + M2.T
    
    C3 = (diag_a[:, None] - a.T) * (Bp.T - diag_Bp[:, None])
    C4 = (a - diag_a[None, :]) * (diag_Bp[None, :] - Bp)
    
    T2 = T2_raw - C3 - C4
    
    # Term 3 matrix
    T3 = (a - a.T) * (Bp.T - Bp)
    
    D = T1 + T2 + T3
    
    # The diagonal should be 0 (swapping with self).
    np.fill_diagonal(D, 0)
    
    return D

def _perturb_permutation(p, k):
    """
    Perturb a permutation by performing k random swaps.
    """
    n = len(p)
    p_new = p.copy()
    for _ in range(k):
        i = np.random.randint(n)
        j = np.random.randint(n)
        p_new[i], p_new[j] = p_new[j], p_new[i]
    return p_new

def _local_search_fast(a, b, p, max_iter=None, time_limit=None, start_time=None):
    """
    Perform steepest descent local search using the full delta matrix.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    
    if max_iter is None:
        max_iter = 1000
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 110.0
    
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iter:
        if time.time() - start_time > time_limit:
            break
            
        D = _compute_delta_matrix(a, b, p)
        
        # Find the best improving move
        min_val = np.min(D)
        
        if min_val < 0:
            # Find the indices of the minimum
            min_idx = np.argmin(D)
            i, j = np.unravel_index(min_idx, D.shape)
            
            # Apply swap
            p[i], p[j] = p[j], p[i]
            current_cost += min_val
            improved = True
            iter_count += 1
        else:
            improved = False
            
    return p, current_cost

def _sa_polish(a, b, p, current_cost, time_limit, start_time):
    """
    Simulated annealing polish with vectorized move selection.
    Uses a geometric cooling schedule and batched random move generation.
    """
    n = len(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # SA parameters
    T_start = max(50.0, current_cost * 0.02)
    T_end = 1.0
    alpha = 0.995  # geometric cooling factor
    
    T = T_start
    batch_size = 100  # number of random moves to generate per batch
    
    while time.time() - start_time < time_limit:
        # Generate batch of random moves
        indices = np.random.randint(0, n, size=(batch_size, 2))
        
        for i, j in indices:
            if i == j:
                continue
            
            delta = _compute_swap_delta(a, b, p, i, j)
            new_cost = current_cost + delta
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                p[i], p[j] = p[j], p[i]
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
        
        # Cool down
        T *= alpha
        if T < T_end:
            T = T_end
    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    total_deadline = 115.0
    start_time = time.time()
    
    # Multi-start: try several different initializations
    best_p = None
    best_cost = float('inf')
    
    # Start 1: Greedy initialization (fast and good)
    p = _greedy_initial(a, b)
    current_cost = _qap_cost(a, b, p)
    
    # Local search to refine
    p, current_cost = _local_search_fast(a, b, p, max_iter=1000, time_limit=total_deadline - 15, start_time=start_time)
    
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # Start 2: Random initialization
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 15:
        p = _random_initial(n)
        current_cost = _qap_cost(a, b, p)
        
        # Local search to refine
        p, current_cost = _local_search_fast(a, b, p, max_iter=1000, time_limit=total_deadline - 15, start_time=start_time)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
    # If we still have time, try additional random starts
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 15:
        # Try more random starts to explore different basins
        num_random_starts = min(8, int(remaining_time // 10))
        for _ in range(num_random_starts):
            if time.time() - start_time > total_deadline - 15:
                break
            
            p = _random_initial(n)
            current_cost = _qap_cost(a, b, p)
            
            # Local search to refine
            p, current_cost = _local_search_fast(a, b, p, max_iter=1000, time_limit=total_deadline - 15, start_time=start_time)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                
    # Final polish with SA on the best solution found
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 10 and best_p is not None:
        p = best_p.copy()
        current_cost = best_cost
        
        # SA polish with improved cooling
        best_p, best_cost = _sa_polish(a, b, p, current_cost, 
                                        time_limit=total_deadline - 2, 
                                        start_time=start_time)
    
    if best_p is None:
        best_p = _greedy_initial(a, b)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
