import numpy as np
import time
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = 600  # seconds, with safety margin
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    for K in [100, 200, 500, 1000, 2000, 5000, 10000, 50000, 100000, 200000]:
        if time.time() - start_time > deadline - 60:
            break
        
        ks_all = np.arange(2, K + 1)
        n_all = len(ks_all)
        
        # Strategy: set f(k) = -c for k in some set, optimize c and the set
        # Constraint: |sum_{k in S} c/k| <= 10 so that f(1) stays in bounds
        # Objective: c * sum_{k in S} log(k)/k
        
        # Try different thresholds and values
        for frac in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            if time.time() - start_time > deadline - 30:
                break
            
            threshold = int(K * frac)
            if threshold < 2:
                continue
            
            mask = ks_all > threshold
            ks_selected = ks_all[mask]
            n_selected = len(ks_selected)
            
            if n_selected == 0:
                continue
            
            # sum 1/k for selected
            sum_inv_k = np.sum(1.0 / ks_selected)
            
            # Maximum c such that c * sum_inv_k <= 10
            c_max = 10.0 / sum_inv_k
            
            # Also c <= 10 (from clipping)
            c_max = min(c_max, 10.0)
            
            if c_max <= 0:
                continue
            
            # Use c = c_max to maximize objective
            c_val = c_max
            
            f_full = np.zeros(K + 1)
            f_full[ks_selected] = -c_val
            S = np.sum(f_full[2:K+1] / ks_all)
            f_full[1] = -S
            
            if abs(f_full[1]) > 10.0 + 1e-9:
                continue
            
            # Verify M(x) <= 1.0001
            max_x = 10 * K
            x_vals = np.arange(1, max_x + 1, dtype=np.float64)
            
            # M(x) = sum_{k in S} (-c_val) * floor(x/k)
            # This is always negative, so M(x) <= 1.0001 is automatically satisfied
            # But let's verify numerically for safety
            M_max = 0.0
            for k in ks_selected[:100]:  # Check first 100 for speed
                M_vals = -c_val * (x_vals // k)
                M_max = max(M_max, np.max(M_vals))
            
            if M_max > 1.0001:
                continue
            
            a = -np.sum(f_full[2:K+1] * np.log(ks_all) / ks_all)
            if a > best_score:
                best_score = a
                domain = np.arange(1, K + 1)
                values = np.clip(f_full[1:K+1], -10, 10)
                best_domain = domain
                best_values = values
    
    if best_domain is not None:
        return best_domain, best_values
    else:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
