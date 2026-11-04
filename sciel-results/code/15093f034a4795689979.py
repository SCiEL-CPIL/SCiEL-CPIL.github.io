"""
QAP solver for wil100 using Frank-Wolfe continuous relaxation followed by
discrete local search and simulated annealing polish.
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
    
    heavy = np.argsort(-a_row_sums, kind="stable")
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

def _compute_all_swap_deltas(a, b, p):
    """
    Compute the full n x n matrix of 2-swap deltas.
    D[i, j] is the cost change if we swap positions i and j.
    """
    n = len(p)
    Bp = b[np.ix_(p, p)]
    
    M1 = a.T @ Bp
    M2 = a @ Bp
    
    diag_M1 = np.diag(M1)
    diag_M2 = np.diag(M2)
    diag_Bp = np.diag(Bp)
    
    T1_raw = M1 - diag_M1[:, None] - diag_M1[None, :] + M1.T
    T2_raw = M2 - diag_M2[:, None] - diag_M2[None, :] + M2.T
    
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

def _frank_wolfe_initialize(a, b, n, time_budget=30.0):
    """
    Use Frank-Wolfe (conditional gradient) on the Birkhoff polytope to find
    a continuous relaxation of the assignment, then round to a permutation.
    """
    start_time = time.time()
    
    # Gradient of QAP cost w.r.t. assignment matrix X:
    # cost = sum_{i,j,k,l} a[i,j] * b[k,l] * X[i,k] * X[j,l]
    # d/dX[i,k] = sum_j a[i,j] * sum_l b[k,l] * X[j,l] + sum_j a[j,i] * sum_l b[l,k] * X[j,l]
    #           = (A @ X @ B)[i,k] + (A.T @ X @ B.T)[i,k]
    
    # Initialize X as a random doubly stochastic matrix (or uniform)
    X = np.ones((n, n)) / n
    
    # Frank-Wolfe iterations
    step_sizes = []
    for it in range(200):
        if time.time() - start_time > time_budget:
            break
        
        # Compute gradient
        G = a @ X @ b + a.T @ X @ b.T
        
        # Find vertex (permutation) minimizing linear objective: G @ X_vertex
        # This is the Hungarian algorithm on cost matrix G
        row_ind, col_ind = linear_sum_assignment(G)
        X_vertex = np.zeros((n, n))
        X_vertex[row_ind, col_ind] = 1.0
        
        # Frank-Wolfe update: X_{t+1} = (1 - gamma) * X_t + gamma * X_vertex
        # Use standard FW step size: gamma = 2 / (t + 2)
        gamma = 2.0 / (it + 2)
        
        # Check if X is close to X_vertex
        if np.sum(np.abs(X - X_vertex)) < 1e-10:
            X = X_vertex.copy()
            break
        
        X = (1 - gamma) * X + gamma * X_vertex
    
    # Round to permutation using Hungarian on the final X
    # Minimize sum of (1 - X[i,j]) for selected pairs, i.e., maximize sum X[i,j]
    row_ind, col_ind = linear_sum_assignment(-X)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return p

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
        D = _compute_all_swap_deltas(a, b, p)
        best_delta = np.min(D)
        best_idx = np.unravel_index(np.argmin(D), D.shape)
        
        if best_delta >= 0:
            break
            
        i, j = best_idx
        p[i], p[j] = p[j], p[i]
        current_cost += int(best_delta)
        
    return p, current_cost

def _sa_polish(a, b, p, current_cost, time_limit, start_time):
    """
    Simulated Annealing polish to escape local minima.
    """
    n = len(p)
    best_p = p.copy()
    best_cost = current_cost
    
    T_start = max(1.0, current_cost * 0.01)
    T_end = 0.05
    elapsed_total = time_limit
    
    while time.time() - start_time < time_limit:
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
    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    total_deadline = 145.0
    start_time = time.time()
    
    best_p = None
    best_cost = float('inf')
    
    # Multiple diverse initializations
    initializations = []
    
    # Frank-Wolfe initialization
    p_fw = _frank_wolfe_initialize(a, b, n, time_budget=30.0)
    initializations.append(p_fw)
    
    # Greedy
    p_greedy = _greedy_initial(a, b)
    initializations.append(p_greedy)
    
    # Random starts
    for _ in range(2):
        initializations.append(_random_initial(n))
    
    # Process each initialization
    for p_init in initializations:
        if time.time() - start_time > total_deadline - 10:
            break
        
        p = p_init.copy()
        current_cost = _qap_cost(a, b, p)
        
        # Local search
        remaining = total_deadline - (time.time() - start_time)
        ls_time = min(remaining * 0.5, 20.0)
        if ls_time > 1:
            p, current_cost = _local_search_fast(a, b, p, time_limit=ls_time, start_time=start_time)
        
        # SA polish
        remaining = total_deadline - (time.time() - start_time)
        sa_time = min(remaining * 0.3, 10.0)
        if sa_time > 1:
            p, current_cost = _sa_polish(a, b, p, current_cost, sa_time, start_time)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    # Final polish on the best solution
    remaining = total_deadline - (time.time() - start_time)
    if remaining > 3 and best_p is not None:
        p = best_p.copy()
        current_cost = best_cost
        
        # One more local search
        p, current_cost = _local_search_fast(a, b, p, time_limit=min(remaining * 0.5, 5.0), start_time=start_time)
        
        # One more SA
        remaining = total_deadline - (time.time() - start_time)
        if remaining > 2:
            p, current_cost = _sa_polish(a, b, p, current_cost, min(remaining * 0.8, 5.0), start_time=start_time)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p
    
    if best_p is None:
        best_p = _greedy_initial(a, b)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
