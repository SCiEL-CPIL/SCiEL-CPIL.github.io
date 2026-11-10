# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 580

    # Fallback to seed result
    def fallback():
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    best_a = 0.0
    best_domain = None
    best_values = None

    # Try different K values
    # The seed program used K up to 100000 but fell back to small K.
    # LP is faster for small K.
    for K in [50, 100, 200, 500]:
        if time.time() > deadline - 10:
            break

        # Domain is 2..K
        domain = np.arange(2, K + 1, dtype=np.int64)
        m = len(domain)
        
        if m == 0:
            continue

        # Construct constraint matrix A and vector b
        # Constraints: for x in 1..10K, sum_k f_k (floor(x/k) - x/k) <= 1.0001
        # Note: floor(x/k) - x/k is negative.
        # Let g_k(x) = floor(x/k) - x/k.
        # We want sum f_k g_k(x) <= 1.0001.
        
        max_x = 10 * K
        # A has shape (max_x, m)
        # We can compute this efficiently.
        # A[x-1, k-2] = floor(x/k) - x/k
        
        # This might be memory intensive for large K.
        # K=500 -> 5000 constraints, 499 variables. 5000*499 ~ 2.5M floats. OK.
        # K=1000 -> 10000 constraints, 999 variables. ~10M floats. OK.
        
        try:
            # Vectorized construction of A
            # x is 1..max_x
            x = np.arange(1, max_x + 1, dtype=np.float64)[:, None] # (max_x, 1)
            k = domain.astype(np.float64)[None, :] # (1, m)
            
            # floor(x/k)
            floor_xk = np.floor(x / k)
            # g = floor(x/k) - x/k
            A = floor_xk - x / k
            
            b = np.ones(max_x, dtype=np.float64) * 1.0001
            
            # Objective: maximize -sum f_k log(k)/k
            # linprog minimizes c^T f
            # So c = - ( - log(k)/k ) = log(k)/k
            c = np.log(domain.astype(np.float64)) / domain.astype(np.float64)
            
            bounds = [(-10.0, 10.0)] * m
            
            res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
            
            if res.status == 0:
                f_vals = res.x
                # Calculate a
                a = -np.sum(f_vals * c)
                
                # Verify feasibility manually to be safe
                # M(x) = sum f_k floor(x/k) + f1 x
                # f1 = - sum f_k / k
                S = np.sum(f_vals / domain.astype(np.float64))
                f1 = -S
                
                # Check max M(x)
                # M(x) = sum_{k in D} f_k floor(x/k) + f1 x
                # We can compute M for all x efficiently
                D = np.zeros(max_x + 2, dtype=np.float64)
                for idx in range(m):
                    k_val = int(domain[idx])
                    fk = f_vals[idx]
                    j = 1
                    while j * k_val <= max_x:
                        lo = j * k_val
                        hi = min((j + 1) * k_val - 1, max_x)
                        D[lo] += fk
                        if hi + 1 <= max_x:
                            D[hi + 1] -= fk
                        j += 1
                M_k_geq_2 = np.cumsum(D)[:max_x + 1]
                M = M_k_geq_2 + f1 * np.arange(max_x + 1, dtype=np.float64)
                M[0] = 0.0
                
                max_M = np.max(M[1:])
                if max_M <= 1.0001 + 1e-6:
                    if a > best_a:
                        best_a = a
                        best_domain = domain.copy()
                        best_values = f_vals.copy()
        except Exception as e:
            print(f"LP failed for K={K}: {e}")
            continue

    if best_domain is not None and best_a > 0.93:
        return best_domain, best_values

    return fallback()

# EVOLVE-BLOCK-END
