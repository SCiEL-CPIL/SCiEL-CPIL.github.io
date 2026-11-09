# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import linprog


def _solve_lp_for_K(K, time_limit=50.0):
    """Solve LP for given K. Returns (domain, values) or None."""
    if K < 2:
        return None
    
    n_vars = K - 1  # g_2, ..., g_K
    k_vals = np.arange(2, K + 1, dtype=np.float64)
    
    # Objective: minimize sum g(k) * log(k)/k  (to maximize -sum g(k)*log(k)/k)
    c = np.log(k_vals) / k_vals
    
    # Constraints: sum_{k=2}^K g(k) * (floor(x/k) - x/k) <= 1.0001 for x in 1..10K
    x_max = 10 * K
    x_vals = np.arange(1, x_max + 1, dtype=np.float64)
    
    # A[i, j] = floor(x_i / k_j) - x_i / k_j
    # Shape: (x_max, n_vars)
    A = np.floor(x_vals[:, None] / k_vals[None, :]) - x_vals[:, None] / k_vals[None, :]
    b = np.full(x_max, 1.0001)
    
    # Bounds: g(k) in [-10, 10]
    bounds = [(-10.0, 10.0)] * n_vars
    
    # Additional constraint: f(1) = -sum g(k)/k must be in [-10, 10]
    # i.e., -10 <= -sum g(k)/k <= 10
    # => -10 <= -sum g(k)/k  =>  sum g(k)/k <= 10
    # => sum g(k)/k <= 10  =>  sum (1/k) * g(k) <= 10
    # => -sum g(k)/k <= 10  =>  sum g(k)/k >= -10  =>  -sum (1/k)*g(k) <= 10
    
    inv_k = 1.0 / k_vals
    # Constraint 1: sum inv_k * g <= 10
    # Constraint 2: -sum inv_k * g <= 10
    A_extra = np.vstack([inv_k, -inv_k])
    b_extra = np.array([10.0, 10.0])
    
    A_full = np.vstack([A, A_extra])
    b_full = np.concatenate([b, b_extra])
    
    try:
        res = linprog(c, A_ub=A_full, b_ub=b_full, bounds=bounds, method='highs')
        if res.status != 0 or not res.success:
            return None
        
        g = res.x
        f1 = -np.sum(g / k_vals)
        
        # Verify f1 is in [-10, 10]
        if f1 < -10.0 - 1e-9 or f1 > 10.0 + 1e-9:
            return None
        
        # Build domain and values
        domain = np.arange(1, K + 1, dtype=np.int64)
        values = np.zeros(K, dtype=np.float64)
        values[0] = f1
        values[1:] = g
        
        return domain, values
    except Exception:
        return None


def _evaluate_result(domain, values, K):
    """Apply evaluator logic and check feasibility. Returns (feasible, a)."""
    # Clip values to [-10, 10]
    clipped = np.clip(values, -10.0, 10.0)
    
    # Build f array for k=1..K
    f = np.zeros(K + 1, dtype=np.float64)
    for i, k in enumerate(domain):
        if 1 <= k <= K:
            f[k] = clipped[i]
    
    # S = sum_k f(k)/k over the clipped values (only where k in domain)
    S = np.sum(clipped / domain.astype(np.float64))
    f[1] = f[1] - S
    
    # M(x) = sum_k f(k) * floor(x/k) for x in 1..10K
    x_max = 10 * K
    x_vals = np.arange(1, x_max + 1, dtype=np.float64)
    k_vals = np.arange(1, K + 1, dtype=np.float64)
    
    # Compute M(x) = sum_k f(k) * floor(x/k)
    # M = floor_matrix @ f[1:]
    floor_mat = np.floor(x_vals[:, None] / k_vals[None, :])
    M = floor_mat @ f[1:]
    
    if np.any(M > 1.0001):
        return False, 0.0
    
    # a = -sum_k f(k) * log(k) / k
    log_k = np.log(k_vals)
    a = -np.sum(f[1:] * log_k / k_vals)
    
    return True, float(a)


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 600s limit
    
    # Seed fallback
    domain_seed = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values_seed = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    best_domain = domain_seed
    best_values = values_seed
    best_score = 1.0  # seed score
    
    # Try LP for increasing K values
    K_list = [50, 100, 200, 300, 500, 800, 1000]
    
    for K in K_list:
        if time.time() > deadline:
            break
        
        # Check if we have enough time for this K
        # Rough estimate: LP time scales with K^2 or K^1.5
        # For K=1000, 10K=10000 constraints, 999 vars. Should be feasible.
        
        result = _solve_lp_for_K(K, time_limit=50.0)
        
        if result is not None:
            dom, val = result
            feasible, a = _evaluate_result(dom, val, K)
            if feasible and a > best_score:
                best_score = a
                best_domain = dom
                best_values = val
        
        if time.time() > deadline - 30.0:
            break
    
    # Final safety check on the best result
    K_best = int(best_domain.max())
    feasible, a_final = _evaluate_result(best_domain, best_values, K_best)
    if not feasible:
        return domain_seed, values_seed
    
    return best_domain, best_values

# EVOLVE-BLOCK-END
