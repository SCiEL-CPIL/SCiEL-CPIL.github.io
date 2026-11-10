import numpy as np
import time
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 600  # Safety margin
    
    best_score = -np.inf
    best_domain = None
    best_values = None
    
    # We will try increasing K
    # K is the maximum key in the domain.
    # We include 1 in the domain, f(1)=0.
    # Variables are f(2)..f(K).
    
    # List of K to try
    K_list = [100, 500, 1000, 2000, 5000]
    
    for K in K_list:
        # Check time
        if time.time() > deadline - 60:
            break
            
        n_vars = K - 1
        if n_vars <= 0:
            continue
            
        # Objective coefficients: c_k = log(k) / k
        ks = np.arange(2, K+1, dtype=np.float64)
        c = np.log(ks) / ks
        
        # Constraints:
        # For each x in [1, 10K]:
        # sum_{k=2}^K f(k) (floor(x/k) - x/k) <= 1
        
        max_x = 10 * K
        
        # Memory check: max_x * n_vars
        # 50000 * 5000 = 250,000,000 floats ~ 2GB
        # 20000 * 2000 = 40,000,000 floats ~ 320MB
        
        try:
            # Construct A
            x_col = np.arange(1, max_x+1, dtype=np.float64)[:, None]
            k_row = ks[None, :]
            
            A = np.floor(x_col / k_row) - x_col / k_row
            b = np.ones(max_x, dtype=np.float64)
            
            # Bounds
            bounds = [(-10.0, 10.0)] * n_vars
            
            # Solve LP
            res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
            
            if res.success:
                f_vars = res.x
                # Calculate score
                a = -np.sum(f_vars * c)
                
                # Construct domain and values
                domain = np.arange(1, K+1, dtype=np.int64)
                values = np.zeros(K, dtype=np.float64)
                values[1:] = f_vars
                values[0] = 0.0 # f(1)
                
                if a > best_score:
                    best_score = a
                    best_domain = domain
                    best_values = values
                    
        except Exception as e:
            # If it fails (e.g. memory), try smaller K or break
            print(f"LP failed for K={K}: {e}")
            # If K is small, we might want to continue to smaller? No, we are increasing.
            # If K=5000 fails, smaller ones probably succeeded.
            continue

    # If no LP solution found, return baseline
    if best_domain is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    else:
        domain = best_domain
        values = best_values

    return domain, values
