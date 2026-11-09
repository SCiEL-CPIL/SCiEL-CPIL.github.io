# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    
    Strategy:
      1. The objective is a = -sum_k f(k) log(k)/k.
      2. The constraint is M(x) = sum_k f(k) floor(x/k) <= 1.0001 for x in [1, 10K].
      3. The evaluator adjusts f(1) so that sum_k f(k)/k = 0.
      4. This implies M(x) = -sum_k f(k) {x/k} for integer x.
      5. Since {x/1} = 0, f(1) does not appear in the constraints.
      6. We can solve an LP for f(2)..f(K) to maximize the objective subject to the constraints.
    """
    start_time = time.time()
    time_limit = 630.0
    safety_margin = 30.0
    
    # Fallback solution
    def get_fallback():
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    # Try increasing K values
    K_candidates = [1000, 2000]
    
    best_domain = None
    best_values = None
    best_score = -np.inf

    for K in K_candidates:
        if time.time() - start_time > time_limit - safety_margin:
            break
            
        try:
            # Variables: f_2 ... f_K
            n_vars = K - 1
            indices = np.arange(2, K + 1)
            
            # Objective: minimize sum f(k) log(k)/k
            c = np.log(indices) / indices
            
            # Constraints: sum_{k=2}^K f(k) {x/k} >= -1.0001 for x in 1..10K
            # linprog form: A_ub @ x <= b_ub
            # So -A @ f <= 1.0001
            # A_{x, k-2} = {x/k}
            
            x_max = 10 * K
            x_vals = np.arange(1, x_max + 1)
            
            # Construct A matrix
            # For K=2000, size is 20000 x 1999. ~320MB.
            # For K=1000, size is 10000 x 999. ~80MB.
            
            # Vectorized construction
            frac_parts = (x_vals[:, np.newaxis] / indices[np.newaxis, :]) % 1.0
            A_ub = -frac_parts
            b_ub = np.full(x_max, 1.0001)
            
            bounds = [(-10.0, 10.0)] * n_vars
            
            # Solve LP
            res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if res.success:
                f_vals = res.x
                # Construct full values
                full_values = np.zeros(K, dtype=np.float64)
                full_values[1:] = f_vals
                
                # Calculate score
                log_k = np.log(np.arange(1, K + 1))
                a = -np.sum(full_values * log_k / np.arange(1, K + 1))
                
                if a > best_score:
                    best_score = a
                    best_domain = np.arange(1, K + 1, dtype=np.int64)
                    best_values = full_values
                    
        except Exception as e:
            continue

    if best_domain is not None:
        return best_domain, best_values
    else:
        return get_fallback()

# EVOLVE-BLOCK-END
