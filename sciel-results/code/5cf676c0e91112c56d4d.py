"""
Improved Simulated Annealing solver for QAPLIB wil100.
Uses multi-start with greedy and Hungarian initializations, fast vectorized
steepest-descent local search, and a tabu-search polish phase.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

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

def _hungarian_initial(a, b):
    """
    Generate an initial permutation using the Hungarian algorithm on a
    linearization of the QAP cost.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Cost[i, j] approximates the marginal cost of assigning facility i to location j
    cost = np.outer(a_row_sums, b_row_sums)
    
    row_ind, col_ind = linear_sum_assignment(cost)
    
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p

def _compute_delta_matrix(a, b, p):
    """
    Compute the full n x n matrix of swap deltas.
    D[i, j] is the cost change if we swap positions i and j.
    Uses O(n^3) matrix multiplications, which is fast for n=100.
    """
    n = len(p)
    Bp = b[np.ix_(p, p)]
    
    # M1[i, j] = sum_k a[k, i] Bp[k, j]
    M1 = a.T @ Bp
    # M2[i, j] = sum_k a[i, k] Bp[k, j]
    M2 = a @ Bp
    
    # Diagonals
    diag_M1 = np.diag(M1)
    diag_M2 = np.diag(M2)
    diag_Bp = np.diag(Bp)
    
    # Raw terms (sum over all k)
    T1_raw = M1 - diag_M1[:, None] - diag_M1[None, :] + M1.T
    T2_raw = M2 - diag_M2[:, None] - diag_M2[None, :] + M2.T
    
    # Corrections for k=i and k=j
    a_diag = np.diag(a)
    
    C1_part1 = (a_diag[:, None] - a) * (Bp - diag_Bp[:, None])
    C1_part2 = (a.T - a_diag[None, :]) * (diag_Bp[None, :] - Bp)
    C1 = C1_part1 + C1_part2
    
    C2_part1 = (a_diag[:, None] - a.T) * (Bp.T - diag_Bp[:, None])
    C2_part2 = (a - a_diag[None, :]) * (diag_Bp[None, :] - Bp.T)
    C2 = C2_part1 + C2_part2
    
    T3 = (a - a.T) * (Bp.T - Bp)
    
    D = (T1_raw - C1) + (T2_raw - C2) + T3
    np.fill_diagonal(D, 0)
    
    return D

def _local_search_fast(a, b, p, time_limit=None, start_time=None):
    """
    Perform steepest descent local search using the vectorized delta matrix.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 110.0
        
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    
    while time.time() - start_time < time_limit:
        D = _compute_delta_matrix(a, b, p)
        best_delta, best_idx = np.min(D), np.unravel_index(np.argmin(D), D.shape)
        
        if best_delta >= 0:
            break
            
        i, j = best_idx
        p[i], p[j] = p[j], p[i]
        current_cost += int(best_delta)
        
    return p, current_cost

def _compute_swap_delta(a, b, p, i, j):
    """
    Compute the cost delta for swapping positions i and j in permutation p.
    Returns the integer delta.
    """
    pi = p[i]
    pj = p[j]
    
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_all_T1 = np.dot(diff_a_col, diff_b_row)
    
    term_1_i = (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    term_1_j = (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    T1 = sum_all_T1 - term_1_i - term_1_j
    
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_all_T2 = np.dot(diff_a_row, diff_b_col)
    
    term_2_i = (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    term_2_j = (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    T2 = sum_all_T2 - term_2_i - term_2_j
    
    T3 = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    return T1 + T2 + T3

def _tabu_search(a, b, p, current_cost, time_limit, start_time):
    """
    Tabu search polish phase.
    """
    n = len(p)
    tabu_list = np.full((n, n), -1, dtype=np.int32)
    tabu_tenure = 20
    best_cost = current_cost
    best_p = p.copy()
    iteration = 0
    
    while time.time() - start_time < time_limit:
        D = _compute_delta_matrix(a, b, p)
        np.fill_diagonal(D, 0)
        
        # Apply tabu: disallow moves that are in the tabu list
        tabu_mask = (iteration - tabu_list) < tabu_tenure
        D_tabu = D.copy()
        D_tabu[tabu_mask] = np.inf
        
        best_delta, best_idx = np.min(D_tabu), np.unravel_index(np.argmin(D_tabu), D_tabu.shape)
        
        if not np.isfinite(best_delta):
            # All moves are tabu; reduce tabu tenure
            tabu_tenure = max(5, tabu_tenure // 2)
            continue
        
        i, j = best_idx
        p[i], p[j] = p[j], p[i]
        tabu_list[i, j] = iteration
        tabu_list[j, i] = iteration
        current_cost += int(best_delta)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        iteration += 1
        
        # Occasionally reset tabu list
        if iteration % 100 == 0:
            tabu_list[:] = -1
    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    total_deadline = 145.0
    start_time = time.time()
    
    best_p = None
    best_cost = float('inf')
    
    # Try multiple initializations
    initializations = []
    
    # Greedy
    p_greedy = _greedy_initial(a, b)
    initializations.append(p_greedy)
    
    # Hungarian
    p_hung = _hungarian_initial(a, b)
    initializations.append(p_hung)
    
    # Random
    for _ in range(3):
        p_rand = _random_initial(n)
        initializations.append(p_rand)
    
    # For each initialization, run local search with a time budget
    time_per_init = (total_deadline - 30) / len(initializations)
    
    for p_init in initializations:
        if time.time() - start_time > total_deadline - 30:
            break
        
        p = p_init.copy()
        current_cost = _qap_cost(a, b, p)
        
        # Local search
        p, current_cost = _local_search_fast(a, b, p, time_limit=total_deadline - 30, start_time=start_time)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    # Tabu search polish on the best solution
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 5 and best_p is not None:
        p = best_p.copy()
        current_cost = best_cost
        p, current_cost = _tabu_search(a, b, p, current_cost, total_deadline - 1, start_time)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    if best_p is None:
        best_p = _greedy_initial(a, b)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
