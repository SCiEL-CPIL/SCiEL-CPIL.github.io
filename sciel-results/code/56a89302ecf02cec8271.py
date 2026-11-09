import numpy as np
import time
import sys
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

def prime_number_theorem_partial_function():
    start_time = time.time()
    timeout = 550  # Safety margin (600s limit)
    
    best_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    best_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    best_score = 1.0

    # Try a range of K values. Smaller K solves faster.
    # Larger K might allow better score but is harder to solve.
    K_candidates = [50, 100, 200, 300, 500]
    
    for K in K_candidates:
        if time.time() - start_time > timeout:
            break
            
        try:
            # Variables f_2 ... f_K
            n_vars = K - 1
            indices = np.arange(2, K + 1, dtype=np.float64)
            
            # Objective: maximize -sum f(k) log(k)/k
            # linprog minimizes, so we minimize sum f(k) log(k)/k
            c = np.log(indices) / indices
            
            # Constraints: sum f(k) * {x/k} >= -1.0001 for x in 1..10K
            # Equivalent to: -sum f(k) * {x/k} <= 1.0001
            # A_ub @ f <= b_ub
            
            x_vals = np.arange(1, 10 * K + 1)
            
            # Construct A_ub efficiently
            # Shape: (10K, K-1)
            # Entry (x, k): -{x/k}
            
            # Vectorized calculation
            # x_vals[:, None] / indices[None, :]
            # This creates a large array. For K=500, 5000x499 ~ 2.5M elements. ~20MB. OK.
            # For K=1000, 10000x999 ~ 10M elements. ~80MB. OK.
            
            frac_parts = (x_vals[:, None] / indices[None, :]) % 1.0
            A_ub = -frac_parts
            
            # Convert to sparse for solver efficiency
            # Most entries are non-zero, so sparse might not reduce memory much,
            # but HiGHS handles sparse input well.
            A_sparse = csr_matrix(A_ub)
            
            b_ub = np.full(10 * K, 1.0001)
            
            bounds = [(-10, 10)] * n_vars
            
            # Solve
            # Use 'highs' for speed and robustness
            res = linprog(c, A_ub=A_sparse, b_ub=b_ub, bounds=bounds, method='highs')
            
            if res.status == 0:
                f_vals = res.x
                
                # Verify feasibility to avoid numerical errors
                # M(x) = -sum(f(k)/k) * x + sum(f(k) * floor(x/k))
                # Or simply: -sum(f(k) * {x/k})
                # Let's use the direct M(x) formula for safety
                S = np.sum(f_vals / indices)
                x_check = np.arange(1, 10 * K + 1)
                # M(x) = -S * x + dot(f_vals, floor(x/k))
                # Using the identity: M(x) = -sum(f(k) * {x/k}) is faster to check?
                # Let's stick to the definition to be sure.
                # floor(x/k) can be computed as x/k - {x/k}
                # M(x) = -S*x + sum f(k)(x/k - {x/k}) = -S*x + x*sum(f(k)/k) - sum(f(k){x/k})
                # Since S = sum(f(k)/k), the first two terms cancel.
                # So M(x) = -sum(f(k) * {x/k})
                
                # We already have frac_parts (which is {x/k})
                # M_vals = -np.dot(f_vals, frac_parts.T)
                # Wait, frac_parts is (10K, K-1). f_vals is (K-1,).
                # M_vals[x] = -sum_k f_vals[k] * frac_parts[x, k]
                M_vals = -np.dot(frac_parts, f_vals)
                
                if np.max(M_vals) <= 1.0001 + 1e-6:
                    # Calculate score
                    score = -np.sum(f_vals * c)
                    if score > best_score:
                        best_score = score
                        # Construct domain and values
                        domain = np.arange(1, K + 1, dtype=np.int64)
                        values = np.zeros(K, dtype=np.float64)
                        values[0] = 0.0 # f(1) orig, evaluator adjusts
                        values[1:] = f_vals
                        best_domain = domain
                        best_values = values
                        # print(f"New best score: {best_score} for K={K}", file=sys.stderr)
                        
        except Exception as e:
            print(f"Error for K={K}: {e}", file=sys.stderr)
            continue
            
    return best_domain, best_values
