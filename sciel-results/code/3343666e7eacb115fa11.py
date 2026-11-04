"""
Hybrid QAP solver for wil100: Riemannian CQAP initialization + vectorized steepest-descent
local search + simulated annealing polish with Lagrangian-relaxation-guided restarts.
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

def _compute_delta_matrix(a, b, p):
    """
    Compute the full n x n matrix of swap deltas.
    D[i, j] is the cost change if we swap positions i and j.
    """
    n = len(p)
    Bp = b[np.ix_(p, p)]
    
    M1 = a.T @ Bp
    M2 = a @ Bp
    
    T1_raw = M1 - np.diag(M1)[:, None] - np.diag(M1)[None, :] + M1.T
    
    diag_a = np.diag(a)
    diag_Bp = np.diag(Bp)
    
    C1 = (diag_a[:, None] - a) * (Bp - diag_Bp[:, None])
    C2 = (a.T - diag_a[None, :]) * (diag_Bp[None, :] - Bp.T)
    
    T1 = T1_raw - C1 - C2
    
    T2_raw = M2 - np.diag(M2)[:, None] - np.diag(M2)[None, :] + M2.T
    
    C3 = (diag_a[:, None] - a.T) * (Bp.T - diag_Bp[:, None])
    C4 = (a - diag_a[None, :]) * (diag_Bp[None, :] - Bp)
    
    T2 = T2_raw - C3 - C4
    
    T3 = (a - a.T) * (Bp.T - Bp)
    
    D = T1 + T2 + T3
    
    np.fill_diagonal(D, 0)
    
    return D

def _riemannian_init(a, b, time_limit=10.0):
    """
    Solve the Continuous Quadratic Assignment Problem (CQAP) using gradient descent
    on the Stiefel manifold to obtain a high-quality initial permutation.
    """
    n = len(a)
    start_time = time.time()
    
    a_norm = a.astype(np.float64)
    b_norm = b.astype(np.float64)
    
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    X = Q.copy()
    
    lr = 0.1
    lr_decay = 0.999
    
    for step in range(500):
        if time.time() - start_time > time_limit:
            break
            
        grad = 2.0 * (a_norm @ X @ b_norm)
        
        sym = (X.T @ grad + grad.T @ X) / 2.0
        grad_tangent = grad - X @ sym
        
        X = X - lr * grad_tangent
        
        Q, R = np.linalg.qr(X)
        d = np.diag(R)
        ph = np.sign(d)
        ph[ph == 0] = 1
        X = Q * ph
        
        lr *= lr_decay
        
        if time.time() - start_time > time_limit:
            break
    
    row_ind, col_ind = linear_sum_assignment(-X)
    
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return p

def _lagrangian_init(a, b, time_limit=15.0):
    """
    Use Lagrangian relaxation to obtain a high-quality initial permutation.
    Decompose the QAP into a linear assignment problem with penalty terms,
    solved iteratively via subgradient optimization.
    """
    n = len(a)
    start_time = time.time()
    
    # Normalize matrices for numerical stability
    a_float = a.astype(np.float64)
    b_float = b.astype(np.float64)
    
    # Initialize dual variables (penalties for distance violations)
    # We'll use a simple subgradient method
    # The Lagrangian relaxation: min sum_i,j a[i,j] x[i,j] * sum_k b[p[i],p[k]] x[k,j]
    # Relax the quadratic coupling via dual variables
    
    # Start with zero dual variables
    dual = np.zeros(n, dtype=np.float64)
    
    # Learning rate for subgradient method
    step_size = 1.0
    step_decay = 0.95
    
    best_p = None
    best_cost = float('inf')
    
    max_iters = 30
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Build the effective cost matrix for the linear assignment problem
        # The linear part comes from the flow matrix
        # The penalty part comes from the dual variables
        # We approximate the QAP objective as:
        # sum_i a[i,:] @ b[:, p[i]] + dual[i] * (some distance measure)
        
        # Compute the "effective" cost for each facility-location pair
        # cost[i, j] = sum_k a[i, k] * b[j, p[k]] + dual[i] * b_row_sums[j]
        # But we don't have p yet, so we use an iterative approach
        
        # For the linear assignment, we use:
        # cost[i, j] = (a @ b)[i, j] + dual[i] * b.sum(axis=1)[j]
        
        # Actually, let's use a simpler formulation:
        # The QAP cost can be written as sum_i (a @ X @ B)[i, i] where X is the assignment matrix
        # Relax this to a linear problem with penalties
        
        # Effective cost matrix:
        # For each facility i and location j, the marginal cost is:
        # sum_k a[i, k] * b[j, k] (if we ignore the permutation coupling)
        # Plus a penalty term based on dual variables
        
        # Let's use: cost[i, j] = (a @ b)[i, j] + dual[i] * b.sum(axis=1)[j]
        
        ab = a_float @ b_float
        b_row_sums = b_float.sum(axis=1)
        
        # Add dual penalty
        cost_matrix = ab + dual[:, None] * b_row_sums[None, :]
        
        # Solve the linear assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        
        # Evaluate the true QAP cost
        current_cost = _qap_cost(a, b, p)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        # Update dual variables using subgradient method
        # The subgradient of the Lagrangian dual is related to the distance
        # For simplicity, use the row sums of b as a proxy
        subgrad = b_row_sums - b_row_sums.mean()
        
        # Update dual variables
        dual += step_size * subgrad / n
        step_size *= step_decay
        
        # Check convergence
        if it > 5 and abs(step_size) < 0.01:
            break
    
    if best_p is None:
        best_p = _greedy_initial(a, b)
    
    return best_p

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
        
        min_val = np.min(D)
        
        if min_val < 0:
            min_idx = np.argmin(D)
            i, j = np.unravel_index(min_idx, D.shape)
            
            p[i], p[j] = p[j], p[i]
            current_cost += min_val
            improved = True
            iter_count += 1
        else:
            improved = False
            
    return p, current_cost

def _simulated_annealing(a, b, p, current_cost, time_limit=30.0, start_time=None):
    """
    Simulated annealing polish.
    """
    n = len(p)
    if start_time is None:
        start_time = time.time()
    
    best_cost = current_cost
    best_p = p.copy()
    
    T_start = max(1.0, current_cost * 0.01)
    T_end = 0.1
    elapsed_total = time_limit
    
    while time.time() - start_time < start_time + time_limit - 1:
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
    total_deadline = 115.0
    start_time = time.time()
    
    best_p = None
    best_cost = float('inf')
    
    # Start 1: Riemannian CQAP initialization
    remaining = total_deadline - (time.time() - start_time)
    if remaining > 20:
        p = _riemannian_init(a, b, time_limit=min(15.0, remaining - 10))
        current_cost = _qap_cost(a, b, p)
        p, current_cost = _local_search_fast(a, b, p, max_iter=500, time_limit=total_deadline - 5, start_time=start_time)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    # Start 2: Lagrangian relaxation initialization
    remaining = total_deadline - (time.time() - start_time)
    if remaining > 20:
        p = _lagrangian_init(a, b, time_limit=min(15.0, remaining - 10))
        current_cost = _qap_cost(a, b, p)
        p, current_cost = _local_search_fast(a, b, p, max_iter=500, time_limit=total_deadline - 5, start_time=start_time)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    # Start 3: Greedy initialization
    remaining = total_deadline - (time.time() - start_time)
    if remaining > 15:
        p = _greedy_initial(a, b)
        current_cost = _qap_cost(a, b, p)
        p, current_cost = _local_search_fast(a, b, p, max_iter=500, time_limit=total_deadline - 5, start_time=start_time)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    # Additional random starts if time permits
    remaining = total_deadline - (time.time() - start_time)
    if remaining > 15:
        num_random_starts = min(3, int(remaining // 15))
        for _ in range(num_random_starts):
            if time.time() - start_time > total_deadline - 10:
                break
            
            p = _random_initial(n)
            current_cost = _qap_cost(a, b, p)
            p, current_cost = _local_search_fast(a, b, p, max_iter=300, time_limit=total_deadline - 5, start_time=start_time)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
    
    # Final SA polish on the best solution
    remaining = total_deadline - (time.time() - start_time)
    if remaining > 5 and best_p is not None:
        p = best_p.copy()
        current_cost = best_cost
        sa_time = min(20.0, remaining - 2)
        p, current_cost = _simulated_annealing(a, b, p, current_cost, time_limit=sa_time, start_time=start_time)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    if best_p is None:
        best_p = _greedy_initial(a, b)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
