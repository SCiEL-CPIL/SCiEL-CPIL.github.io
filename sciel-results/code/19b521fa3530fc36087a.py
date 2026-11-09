# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit
    
    # Use K to balance LP size vs. objective quality
    K = 500
    
    # Variables: g(2), g(3), ..., g(K)
    n_vars = K - 1  # k = 2, 3, ..., K
    
    ks = np.arange(2, K + 1)
    
    # Objective: minimize sum g(k)*log(k)/k  (equivalent to maximizing -sum g(k)*log(k)/k)
    c = np.log(ks) / ks
    
    # Constraints: M(x) = sum_{k=2}^{K} g(k) * (x/k + floor(x/k)) <= 1.0001 for x in [1, 10K]
    X_max = 10 * K
    A_ub = np.zeros((X_max, n_vars))
    for i, x in enumerate(range(1, X_max + 1)):
        for j, k in enumerate(ks):
            A_ub[i, j] = x / k + np.floor(x / k)
    
    b_ub = np.full(X_max, 1.0001)
    
    # Bounds: -10 <= g(k) <= 10
    bounds = [(-10.0, 10.0)] * n_vars
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.success:
        g_vals = result.x  # g(2), g(3), ..., g(K)
        
        # Compute f(1) = S = sum g(k)/k
        S = np.sum(g_vals / ks)
        
        # Build domain and values
        domain = np.concatenate(([1], ks))
        values = np.concatenate(([S], g_vals))
        
        # Clip values to [-10, 10]
        values = np.clip(values, -10.0, 10.0)
        
        # Verify feasibility: check M(x) for all x in [1, 10K]
        f_dict = dict(zip(domain, values))
        max_M = 0.0
        for x in range(1, X_max + 1):
            Mx = 0.0
            for k, fk in f_dict.items():
                Mx += fk * (x // k)
            if Mx > max_M:
                max_M = Mx
        
        if max_M > 1.0001:
            # Infeasible, return fallback
            domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
            values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
            return domain, values
        
        return domain.astype(np.int64), values.astype(np.float64)
    else:
        # Fallback
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

# EVOLVE-BLOCK-END
