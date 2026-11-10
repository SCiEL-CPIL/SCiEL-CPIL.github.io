# EVOLVE-BLOCK-START
import numpy as np
import time


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
    deadline = start_time + 600
    
    def solve_lp(K, tol=1e-8):
        """
        Solve the LP for a given K with tolerance tol.
        Returns (domain, values, a) or None if infeasible.
        """
        n_vars = K - 1
        k_vals = np.arange(2, K+1)
        
        # Objective: minimize sum g(k) log(k)/k
        c = np.log(k_vals) / k_vals
        
        # Constraints: sum g(k) h_k(x) <= 1 for x in 1..10K
        # h_k(x) = floor(x/k) - x/k
        # Compute efficiently using integer division
        x_max = 10 * K
        x_vals = np.arange(1, x_max + 1)
        
        # Build constraint matrix A_ub of shape (x_max, n_vars)
        # A_ub[i, j] = floor(x_vals[i] / k_vals[j]) - x_vals[i] / k_vals[j]
        A_ub = np.zeros((x_max, n_vars), dtype=np.float64)
        for j, k in enumerate(k_vals):
            A_ub[:, j] = np.floor(x_vals / k) - x_vals / k
        
        b_ub = np.ones(x_max)
        bounds = [(-10, 10)] * n_vars
        
        from scipy.optimize import linprog
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', 
                     options={'presolve': True, 'dual_feasibility_tolerance': tol,
                              'primal_feasibility_tolerance': tol})
        
        if not res.success:
            return None
        
        g = res.x
        # Construct f
        f = np.zeros(K+1)
        f[2:K+1] = g
        S = np.sum(f[2:K+1] / k_vals)
        f[1] = -S
        
        # Simulate evaluator: clip, then adjust f(1)
        f_clipped = np.clip(f, -10, 10)
        S_clip = np.sum(f_clipped[1:K+1] / np.arange(1, K+1))
        f_clipped[1] -= S_clip
        
        # Check feasibility
        domain = np.arange(1, K+1)
        values = f_clipped[1:K+1]
        
        # Calculate M(x) for x in 1..10K
        F = np.floor(x_vals[:, None] / np.arange(1, K+1)[None, :])
        M = F @ values
        
        if np.any(M > 1.0001):
            return None
        
        # Calculate objective
        all_k = np.arange(1, K+1)
        a = -np.sum(values * np.log(all_k) / all_k)
        
        return domain, values, a

    best_result = None
    best_a = -np.inf
    
    # Try increasing K with time checks
    K_list = [10, 20, 50, 100, 200, 500, 1000]
    
    for K in K_list:
        if time.time() > deadline - 120:
            break
        
        res = solve_lp(K)
        if res is not None:
            domain, values, a = res
            if a > best_a:
                best_a = a
                best_result = (domain, values)
        
        # If we have a good result, try to polish with tighter tolerance
        if best_result is not None and best_a > 0.5 and time.time() < deadline - 60:
            # Try a polish pass on the best K found so far
            best_K = len(best_result[0])
            polished = solve_lp(best_K, tol=1e-10)
            if polished is not None:
                p_domain, p_values, p_a = polished
                if p_a > best_a:
                    best_a = p_a
                    best_result = (p_domain, p_values)
    
    if best_result is not None:
        return best_result
    else:
        # Fallback to baseline
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values


# EVOLVE-BLOCK-END
