import numpy as np
import time
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 550  # 550s limit, leaving margin for safety
    
    best_domain = np.array([2, 3, 5], dtype=np.int64)
    best_values = np.array([-1.0, -1.0, 1.0], dtype=np.float64)
    best_score = 0.0 # Start low, seed is 1.0? "1.0 equals the seed program".
    # Actually, let's calculate the score of the initial guess to be safe.
    # But we will update best_score with calculated scores.
    
    # Try increasing K
    K_values = [10, 20, 50, 100, 200, 500, 1000]
    
    for K in K_values:
        if time.time() > deadline - 10:
            break
        
        try:
            # Variables f(2) ... f(K)
            k = np.arange(2, K + 1)
            n_vars = len(k)
            
            # Objective: Minimize sum f(k) * log(k) / k
            # a = - sum f(k) * log(k) / k
            # Maximize a => Minimize sum f(k) * log(k) / k
            c = np.log(k) / k
            
            # Constraints:
            # For each x in [1, 10*K]:
            # sum_{k=2}^K f(k) * (floor(x/k) - x/k) <= 1.0001
            # Note: floor(x/k) - x/k is negative.
            # Let A_ub[x_idx, k_idx] = floor(x/k) - x/k
            # A_ub * f <= b_ub
            
            x = np.arange(1, 10 * K + 1)
            n_cons = len(x)
            
            # Efficient construction of A_ub
            # A_ub[i, j] = x[i] // k[j] - x[i] / k[j]
            # We can compute this in chunks or vectorized.
            # x is (10K,), k is (K,)
            # Result is (10K, K)
            
            # To avoid memory blowup for large K (e.g., K=1000 -> 10000x1000 = 10M floats = 80MB),
            # it's okay. But let's be careful.
            
            A_ub = np.empty((n_cons, n_vars), dtype=np.float64)
            
            # Vectorized calculation
            # x_col = x[:, None]
            # k_row = k[None, :]
            # A_ub = x_col // k_row - x_col / k_row
            
            # This creates two (10K, K) arrays.
            # For K=1000, 10000x1000 = 10M. 80MB each. Total 160MB. Acceptable.
            
            x_col = x[:, None]
            k_row = k[None, :]
            
            # Calculate floor part and fractional part difference
            # floor(x/k) - x/k = - (x/k - floor(x/k)) = - frac(x/k)
            # But we need exact float.
            # Using modulo might be faster or more stable?
            # x % k gives remainder r. x/k = (x-r)/k + r/k = floor + frac.
            # So floor - x/k = -r/k.
            
            # r = x[:, None] % k[None, :]
            # A_ub = -r / k[None, :]
            
            r = x_col % k_row
            A_ub = -r / k_row
            
            b_ub = np.full(n_cons, 1.0001, dtype=np.float64)
            
            bounds = [(-10.0, 10.0)] * n_vars
            
            # Solve LP
            # Use 'highs' which is fast
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options={'presolve': True})
            
            if res.status == 0:
                f = res.x
                # Filter out small values to keep domain small?
                # The task allows up to 200000 keys.
                # However, returning all K keys is fine.
                # But if many are 0, we can drop them.
                mask = np.abs(f) > 1e-6
                if not np.any(mask):
                    continue
                
                domain_k = k[mask]
                values_f = f[mask]
                
                # Calculate score a
                # a = -sum f(k) log(k) / k
                score = -np.sum(values_f * np.log(domain_k) / domain_k)
                
                if score > best_score:
                    best_score = score
                    best_domain = domain_k
                    best_values = values_f
                    
        except Exception as e:
            print(f"Error for K={K}: {e}")
            continue
            
    return best_domain, best_values
