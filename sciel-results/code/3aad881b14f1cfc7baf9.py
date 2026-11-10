import numpy as np
from scipy.optimize import linprog
import math
import time

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = 600  # seconds
    
    # Strategy: Use LP to find optimal f(k) for k=2..K with constraints.
    # The key insight is that we need M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for all x in [1, 10K].
    # Also, after adjusting f(1), we need sum_k f(k)/k = 0.
    # f(1) is set to -sum_{k>=2} f(k)/k.
    # The metric is a = -sum_k f(k) log(k) / k.
    # Note: f(1) log(1)/1 = 0, so the metric only depends on f(2)..f(K).
    # But wait, the sum for S includes f(1)/1 = f(1).
    # S = sum_{k=1}^K f(k)/k. We set f(1) = -S_rest where S_rest = sum_{k=2}^K f(k)/k.
    # Then S = f(1) + S_rest = 0.
    # The metric a = -sum_{k=1}^K f(k) log(k)/k = -f(1)*0 - sum_{k=2}^K f(k) log(k)/k = -sum_{k=2}^K f(k) log(k)/k.
    # So we just maximize -sum_{k=2}^K f(k) log(k)/k subject to M(x) <= 1.0001.
    
    # Let's try a few K values. Larger K allows more degrees of freedom.
    # However, LP size grows as O(K^2) for constraints (10K constraints, K variables).
    # K=2000 might be too slow for high precision, but let's try.
    
    best_result = None
    best_score = -np.inf
    
    # We'll try K = 1000 and K = 2000 if time permits.
    # Actually, let's start with K=500 which is manageable.
    
    for K in [500, 1000, 2000]:
        if time.time() - start_time > deadline - 120:
            break
            
        n_vars = K - 1
        
        # Objective: maximize - sum f(k) log(k)/k
        # linprog minimizes c^T x, so c = log(k)/k
        c = np.array([math.log(k) / k for k in range(2, K + 1)])
        
        # Constraints: M(x) = sum_{k=2}^K f(k) floor(x/k) <= 1.0001
        # Note: f(1) does not appear in M(x) because floor(x/1) = x, but wait.
        # M(x) = sum_{k=1}^K f(k) floor(x/k).
        # f(1) = -sum_{k=2}^K f(k)/k.
        # So M(x) = f(1)*x + sum_{k=2}^K f(k) floor(x/k)
        # = -x * sum_{k=2}^K f(k)/k + sum_{k=2}^K f(k) floor(x/k)
        # = sum_{k=2}^K f(k) (floor(x/k) - x/k)
        
        # So the constraint is sum_{k=2}^K f(k) (floor(x/k) - x/k) <= 1.0001
        # This matches the previous formulation but with 1.0001 instead of 1.
        
        x_vals = np.arange(1, 10 * K + 1)
        k_vals = np.arange(2, K + 1)
        
        # A_ub[x_idx, k_idx] = floor(x/k) - x/k
        # Shape: (10K, K-1)
        x_col = x_vals[:, np.newaxis]  # (10K, 1)
        k_row = k_vals[np.newaxis, :]  # (1, K-1)
        
        A_ub = np.floor(x_col / k_row) - x_col / k_row  # (10K, K-1)
        b_ub = np.ones(10 * K) * 1.0001
        
        # Bounds
        bounds = [(-10, 10)] * n_vars
        
        # Solve LP
        try:
            # Use highs method for speed
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options={'presolve': True})
            
            if res.success:
                f_vals = res.x
                # Compute the objective value
                obj_val = -np.sum(f_vals * c)
                
                if obj_val > best_score:
                    domain = np.arange(1, K + 1, dtype=np.int64)
                    values = np.zeros(K, dtype=np.float64)
                    values[1:] = f_vals
                    best_result = (domain, values)
                    best_score = obj_val
                    
        except Exception as e:
            print(f"LP failed for K={K}: {e}")
            continue
    
    if best_result is None:
        # Fallback: simple mu-based approach
        K = 100
        mu = np.zeros(K + 1, dtype=int)
        mu[1] = 1
        for i in range(2, K + 1):
            if mu[i] == 0:
                for j in range(i, K + 1, i):
                    mu[j] = -mu[j]
                    if j % (i * i) == 0:
                        mu[j] = 0
        domain = np.arange(1, K + 1, dtype=np.int64)
        values = np.zeros(K, dtype=np.float64)
        values[1:] = mu[2:K + 1].astype(np.float64)
        return domain, values
    
    return best_result
