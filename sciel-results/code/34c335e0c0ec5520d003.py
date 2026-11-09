# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    The function f is scored by a fixed evaluator (see evaluator.py) that:
      * subtracts a constant from f(1) so that sum_k f(k)/k == 0,
      * requires the step function M(x) = sum_k f(k) * floor(x/k) to stay <= 1 for every
        real x in [1, 10*max_key] (otherwise the construction is infeasible), and
      * returns the objective a = -sum_k f(k) * log(k) / k (to be maximized; optimum 1.0).
    The optimum is approached by the Mobius function mu(k), for which the classic identity
    sum_{k<=x} mu(k) floor(x/k) = 1 holds and -sum mu(k) log(k)/k -> 1.

    Representation:
        domain : np.ndarray, shape (m,), dtype int   -> positive integers in the domain of f
        values : np.ndarray, shape (m,), dtype float -> the corresponding real values f(k)

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f.
    """
    start_time = time.time()
    deadline = start_time + 590  # 10 minutes minus safety margin
    
    # We solve a Linear Programming problem to find the optimal f(k) for k in [2, K].
    # The constraint M(x) <= 1 simplifies to sum_{k=2}^K f(k) * (floor(x/k) - x/k) <= 1.
    # The objective is to maximize -sum_{k=2}^K f(k) * log(k) / k.
    
    K = 100  # Increased K for better approximation
    
    if time.time() > deadline - 10:
         # Fallback to seed if time is critically low
         return np.array([1, 2, 3, 5, 30], dtype=np.int64), np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)

    # Variables: f_2, ..., f_K
    n_vars = K - 1
    ks = np.arange(2, K + 1)
    
    # Objective: minimize c^T x where c_k = -log(k)/k
    # We want to maximize sum f_k log(k)/k, so we minimize -sum f_k log(k)/k
    c = -np.log(ks) / ks
    
    # Constraints: A_ineq @ x <= b
    # For each x in [1, 10K], sum_{k=2}^K f_k (floor(x/k) - x/k) <= 1.0001
    max_x = 10 * K
    x_vals = np.arange(1, max_x + 1)
    
    # Construct A_ineq
    # A_ineq has shape (max_x, n_vars)
    # A_ineq[i, j] corresponds to x = x_vals[i] and k = ks[j]
    # term = floor(x/k) - x/k
    
    A_ineq = np.zeros((max_x, n_vars))
    for j, k in enumerate(ks):
        A_ineq[:, j] = np.floor(x_vals / k) - (x_vals / k)
        
    b = np.ones(max_x) * 1.0001
    
    # Bounds for f_k: [-10, 10]
    bounds = [(-10, 10)] * n_vars
    
    # Solve LP
    try:
        res = linprog(c, A_ub=A_ineq, b_ub=b, bounds=bounds, method='highs')
        
        if res.status == 0:
            f_vals = res.x
            domain = ks
            values = f_vals
            return domain, values
        else:
            # If LP fails, fall back to a simple heuristic or seed
            pass
    except Exception:
        pass
    
    # Fallback: Seed values
    return np.array([1, 2, 3, 5, 30], dtype=np.int64), np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)

# EVOLVE-BLOCK-END
