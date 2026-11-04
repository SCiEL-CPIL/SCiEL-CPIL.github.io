"""
Simulated Annealing solver for QAPLIB wil100.
Combines multi-start greedy+random initialization with a highly optimized
steepest-descent local search using vectorized delta matrix computation,
followed by SA/tabu polish phases.
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
    
    # Term 1 matrix: T1[i, j] = M1[i, j] - M1[i, i] - M1[j, j] + M1[j, i]
    # But we need to exclude k=i and k=j from the sum.
    # Let's compute the raw sum first, then subtract corrections.
    
    # Raw T1[i, j] = sum_k a[k,i] b[p[k],p[j]] - sum_k a[k,i] b[p[k],p[i]]
    #               - sum_k a[k,j] b[p[k],p[j]] + sum_k a[k,j] b[p[k],p[i]]
    # = M1[i, j] - M1[i, i] - M1[j, j] + M1[j, i]
    
    T1_raw = M1 - np.diag(M1)[:, None] - np.diag(M1)[None, :] + M1.T
    
    # Corrections for T1:
    # k=i: (a[i,i] - a[i,j]) * (b[p[i],p[j]] - b[p[i],p[i]])
    #      = (a[i,i] - a[i,j]) * (Bp[i,j] - Bp[i,i])
    # k=j: (a[j,i] - a[j,j]) * (b[p[j],p[j]] - b[p[j],p[i]])
    #      = (a[j,i] - a[j,j]) * (Bp[j,j] - Bp[j,i])
    
    # We need to construct a matrix of corrections.
    # corr_i[i, j] = (a[i,i] - a[i,j]) * (Bp[i,j] - Bp[i,i])
    # This is hard to vectorize as a full matrix directly without loops or advanced indexing.
    # Let's try to vectorize it.
    # a[i,i] is diag(a).
    # a[i,j] is a.
    # Bp[i,j] is Bp.
    # Bp[i,i] is diag(Bp).
    
    diag_a = np.diag(a)
    diag_Bp = np.diag(Bp)
    
    # corr_i[i, j] = (diag_a[i] - a[i,j]) * (Bp[i,j] - diag_Bp[i])
    # Let's compute this element-wise.
    # (diag_a[:, None] - a) * (Bp - diag_Bp[:, None])
    # This gives a matrix C1 where C1[i, j] = (a[i,i] - a[i,j]) * (Bp[i,j] - Bp[i,i])
    C1 = (diag_a[:, None] - a) * (Bp - diag_Bp[:, None])
    
    # corr_j[i, j] = (a[j,i] - a[j,j]) * (Bp[j,j] - Bp[j,i])
    # Note that corr_j[i, j] is the same as corr_j.T[j, i] if we define it carefully?
    # Let's define C2[i, j] = (a[j,i] - a[j,j]) * (Bp[j,j] - Bp[j,i])
    # This is equivalent to:
    # a[j,i] is a.T[i, j]
    # a[j,j] is diag_a[j]
    # Bp[j,j] is diag_Bp[j]
    # Bp[j,i] is Bp.T[i, j] (Bp is symmetric)
    
    # So C2[i, j] = (a.T[i, j] - diag_a[j]) * (diag_Bp[j] - Bp.T[i, j])
    # Let's compute C2.
    C2 = (a.T - diag_a[None, :]) * (diag_Bp[None, :] - Bp.T)
    
    T1 = T1_raw - C1 - C2
    
    # Term 2 matrix: T2[i, j] = M2[i, j] - M2[i, i] - M2[j, j] + M2[j, i]
    # Wait, M2[i, j] = sum_k a[i, k] b[p[k], p[j]]
    # T2[i, j] = sum_k (a[i,k] - a[j,k]) * (b[p[j],p[k]] - b[p[i],p[k]])
    # = sum_k a[i,k] b[p[j],p[k]] - sum_k a[i,k] b[p[i],p[k]]
    #   - sum_k a[j,k] b[p[j],p[k]] + sum_k a[j,k] b[p[i],p[k]]
    # = M2[i, j] - M2[i, i] - M2[j, j] + M2[j, i]
    
    T2_raw = M2 - np.diag(M2)[:, None] - np.diag(M2)[None, :] + M2.T
    
    # Corrections for T2:
    # k=i: (a[i,i] - a[j,i]) * (b[p[j],p[i]] - b[p[i],p[i]])
    #      = (a[i,i] - a[j,i]) * (Bp[j,i] - Bp[i,i])
    # k=j: (a[i,j] - a[j,j]) * (b[p[j],p[j]] - b[p[i],p[j]])
    #      = (a[i,j] - a[j,j]) * (Bp[j,j] - Bp[i,j])
    
    # C3[i, j] = (a[i,i] - a[j,i]) * (Bp[j,i] - Bp[i,i])
    # a[i,i] is diag_a[i]
    # a[j,i] is a.T[i, j]
    # Bp[j,i] is Bp.T[i, j]
    # Bp[i,i] is diag_Bp[i]
    
    C3 = (diag_a[:, None] - a.T) * (Bp.T - diag_Bp[:, None])
    
    # C4[i, j] = (a[i,j] - a[j,j]) * (Bp[j,j] - Bp[i,j])
    # a[i,j] is a[i, j]
    # a[j,j] is diag_a[j]
    # Bp[j,j] is diag_Bp[j]
    # Bp[i,j] is Bp[i, j]
    
    C4 = (a - diag_a[None, :]) * (diag_Bp[None, :] - Bp)
    
    T2 = T2_raw - C3 - C4
    
    # Term 3 matrix: T3[i, j] = (a[i,j] - a[j,i]) * (b[p[j],p[i]] - b[p[i],p[j]])
    # = (a[i,j] - a[j,i]) * (Bp[j,i] - Bp[i,j])
    T3 = (a - a.T) * (Bp.T - Bp)
    
    D = T1 + T2 + T3
    
    # The diagonal should be 0 (swapping with self).
    np.fill_diagonal(D, 0)
    
    return D

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
        # We only care about i < j to avoid duplicates, but D is symmetric?
        # D[i, j] = D[j, i] because swapping i and j is the same as swapping j and i.
        # Let's check.
        # The formula is symmetric in i and j?
        # T1[i, j] = T1[j, i]?
        # M1[i, j] - M1[i, i] - M1[j, j] + M1[j, i]
        # M1[j, i] - M1[j, j] - M1[i, i] + M1[i, j]
        # Yes, it is symmetric.
        # So we can just find the global minimum.
        
        min_val = np.min(D)
        
        if min_val < 0:
            # Find the indices of the minimum
            # There might be multiple, pick the first one
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

def solve(a, b):
    n = len(a)
    total_deadline = 115.0
    start_time = time.time()
    
    # Multi-start: try several different initializations
    best_p = None
    best_cost = float('inf')
    
    # Start 1: Greedy initialization
    p = _greedy_initial(a, b)
    current_cost = _qap_cost(a, b, p)
    
    # Local search to refine
    p, current_cost = _local_search_fast(a, b, p, max_iter=500, time_limit=total_deadline - 5, start_time=start_time)
    
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # If we still have time, try additional random starts
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 10:
        # Try a few random starts
        num_random_starts = min(5, int(remaining_time // 10))
        for _ in range(num_random_starts):
            if time.time() - start_time > total_deadline - 5:
                break
            
            p = _random_initial(n)
            current_cost = _qap_cost(a, b, p)
            
            # Local search to refine
            p, current_cost = _local_search_fast(a, b, p, max_iter=500, time_limit=total_deadline - 5, start_time=start_time)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
    
    # Final polish with SA on the best solution found
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 2 and best_p is not None:
        p = best_p.copy()
        current_cost = best_cost
        
        T_start = max(1.0, best_cost * 0.01)
        T_end = 0.1
        elapsed_total = total_deadline
        iters = 0
        
        while time.time() - start_time < total_deadline - 1:
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            
            delta = _compute_swap_delta(a, b, p, i, j)
            new_cost = current_cost + delta
            
            elapsed = time.time() - start_time
            T = T_start * (T_end / T_start) ** (elapsed / elapsed_total)
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                p[i], p[j] = p[j], p[i]
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            
            iters += 1
    
    if best_p is None:
        best_p = _greedy_initial(a, b)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
