"""Improved QAP solver using Frank-Wolfe on Birkhoff polytope with 2-opt polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    n = len(p)
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))

def _rank_matching_init(a, b):
    """Initial permutation based on rank matching."""
    n = len(a)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def _random_smoothed_init(a, b, rng):
    """Random initialization with some smoothing."""
    n = len(a)
    # Start with rank matching, then perturb
    p = _rank_matching_init(a, b).copy()
    # Random swaps to diversify
    n_swaps = n // 8
    for _ in range(n_swaps):
        i, j = rng.choice(n, 2, replace=False)
        p[i], p[j] = p[j], p[i]
    return p

def _frank_wolfe_qap(a, b, p0, max_iter=500, tol=1e-4, time_limit=None, start_time=None):
    """
    Frank-Wolfe method on the Birkhoff polytope for QAP.
    We maintain a doubly stochastic matrix X and project via Hungarian.
    The QAP objective in matrix form: f(X) = sum_{i,j,k,l} a[i,j] * b[k,l] * X[i,k] * X[j,l]
    This is a quadratic function of X.
    
    Gradient: grad[i,k] = 2 * sum_j a[i,j] * (b[:, k] @ X[j, :]) ... let me derive properly.
    
    f(X) = trace(A^T X B^T X^T) ... actually let's use:
    f(X) = sum_{i,j} a[i,j] * sum_{k,l} b[k,l] * X[i,k] * X[j,l]
    
    Let's define: for fixed X, grad_{i,k} = 2 * sum_j a[i,j] * (sum_l b[k,l] * X[j,l])
    = 2 * (A @ (B @ X))[i,k] ... wait, let me be careful.
    
    f(X) = sum_{i,j,k,l} a[i,j] b[k,l] X[i,k] X[j,l]
    
    df/dX[i,k] = sum_{j,l} a[i,j] b[k,l] X[j,l] + sum_{j,l} a[j,i] b[l,k] X[j,l]
                = (A @ (B @ X))[i,k] + (A.T @ (B.T @ X))[i,k]
    
    For symmetric a and b (which QAP instances often are), this simplifies.
    But let's handle the general case.
    """
    n = len(a)
    if start_time is None:
        start_time = time.time()
    
    # Initialize X as the permutation matrix of p0
    X = np.zeros((n, n), dtype=np.float64)
    X[np.arange(n), p0] = 1.0
    
    best_p = p0.copy()
    best_cost = _qap_cost(a, b, p0)
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Compute gradient
        # grad[i,k] = sum_{j,l} a[i,j] b[k,l] X[j,l] + sum_{j,l} a[j,i] b[l,k] X[j,l]
        # = (A @ (B @ X))[i,k] + (A.T @ (B.T @ X))[i,k]
        BX = b @ X  # shape (n, n)
        ATBX = a.T @ BX  # shape (n, n)
        BXT = b.T @ X  # shape (n, n)
        ATBXT = a.T @ BXT  # shape (n, n)
        # Wait, let me redo this.
        # term1[i,k] = sum_j a[i,j] * (B @ X)[j,k] = (A @ (B @ X))[i,k]
        # term2[i,k] = sum_j a[j,i] * (B.T @ X)[j,k] = (A.T @ (B.T @ X))[i,k]
        
        term1 = a @ (b @ X)
        term2 = a.T @ (b.T @ X)
        grad = term1 + term2
        
        # Frank-Wolfe step: find s = argmin_{S in Birkhoff} <grad, S>
        # This is the Hungarian problem on the matrix grad
        row_ind, col_ind = linear_sum_assignment(grad)
        S = np.zeros((n, n), dtype=np.float64)
        S[row_ind, col_ind] = 1.0
        
        # Line search: find best alpha in [0, 1]
        # f(X + alpha*(S - X)) is quadratic in alpha
        # We can do a simple line search
        best_alpha = 0.0
        best_val = None
        # Sample alphas
        alphas = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        for alpha in alphas:
            X_new = X + alpha * (S - X)
            # Clip to ensure feasibility (should already be in Birkhoff since convex combo)
            # Compute cost
            cost = np.sum(a * (X_new @ b @ X_new.T))
            if best_val is None or cost < best_val:
                best_val = cost
                best_alpha = alpha
        
        X_new = X + best_alpha * (S - X)
        
        # Project to nearest permutation (Hungarian) to get a valid assignment
        row_ind, col_ind = linear_sum_assignment(-X_new)  # maximize sum of X_new
        p_new = col_ind
        
        cost_new = _qap_cost(a, b, p_new)
        if cost_new < best_cost:
            best_cost = cost_new
            best_p = p_new.copy()
        
        X = X_new
        
        # Check convergence
        if it > 10 and abs(best_val - best_cost) < tol:
            break
    
    return best_p, best_cost

def _local_search_2opt(a, b, p, time_limit=None, start_time=None):
    """2-opt local search: try all pairs of swaps, accept if improves."""
    n = len(a)
    if start_time is None:
        start_time = time.time()
    
    current_cost = _qap_cost(a, b, p)
    p = p.copy()
    improved = True
    
    while improved:
        improved = False
        for i in range(n):
            if time_limit is not None and (time.time() - start_time) > time_limit:
                return p, current_cost
            for j in range(i + 1, n):
                # Try swapping p[i] and p[j]
                p[i], p[j] = p[j], p[i]
                new_cost = _qap_cost(a, b, p)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                else:
                    p[i], p[j] = p[j], p[i]
    
    return p, current_cost

def solve(a, b):
    """
    Solve the QAP instance using Frank-Wolfe on Birkhoff polytope with 2-opt polish.
    """
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Convert to int64 if needed
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    
    best_p = None
    best_cost = np.inf
    
    # Multiple restarts with different initializations
    rng = np.random.RandomState(42)
    
    # Restart 1: rank matching
    p0 = _rank_matching_init(a, b)
    p_fw, cost_fw = _frank_wolfe_qap(a, b, p0, max_iter=300, time_limit=30, start_time=start_time)
    if cost_fw < best_cost:
        best_cost = cost_fw
        best_p = p_fw.copy()
    
    if time.time() > deadline:
        return best_p.tolist() if best_p is not None else p0.tolist()
    
    # Restart 2: random smoothed
    p0 = _random_smoothed_init(a, b, rng)
    p_fw, cost_fw = _frank_wolfe_qap(a, b, p0, max_iter=300, time_limit=30, start_time=start_time)
    if cost_fw < best_cost:
        best_cost = cost_fw
        best_p = p_fw.copy()
    
    if time.time() > deadline:
        return best_p.tolist() if best_p is not None else p0.tolist()
    
    # Restart 3: another random
    p0 = _random_smoothed_init(a, b, rng)
    p_fw, cost_fw = _frank_wolfe_qap(a, b, p0, max_iter=300, time_limit=30, start_time=start_time)
    if cost_fw < best_cost:
        best_cost = cost_fw
        best_p = p_fw.copy()
    
    if time.time() > deadline:
        return best_p.tolist() if best_p is not None else p0.tolist()
    
    # Final 2-opt polish on the best solution
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        p_polished, cost_polished = _local_search_2opt(a, b, best_p, time_limit=remaining_time - 5, start_time=time.time())
        if cost_polished < best_cost:
            best_cost = cost_polished
            best_p = p_polished
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
