import numpy as np
import time
from scipy.optimize import linprog

def _compute_mobius(K):
    mu = np.zeros(K + 1, dtype=np.int8)
    mu[1] = 1
    primes = []
    is_composite = np.zeros(K + 1, dtype=bool)
    for i in range(2, K + 1):
        if not is_composite[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_composite[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 600.0 - 10.0
    
    def solve_lp_for_K(K_max):
        """Solve LP for given K_max, return (domain, values) or None."""
        ks = np.arange(2, K_max + 1)
        m = len(ks)
        
        # Objective: maximize -sum f_k log(k)/k
        c = -np.log(ks) / ks
        c_obj = -c
        
        # Constraints: -sum f_k {x/k} <= 1 for x in 1..10K_max
        max_x = 10 * K_max
        A_ub_list = []
        b_ub_list = []
        
        for x in range(1, max_x + 1):
            frac_parts = np.array([(x / k) % 1 for k in ks])
            A_ub_list.append(-frac_parts)
            b_ub_list.append(1.0)
            
        A_ub = np.array(A_ub_list)
        b_ub = np.array(b_ub_list)
        
        # Bounds: -10 <= f_k <= 10
        bounds = [(-10, 10)] * m
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                f_vals = np.clip(res.x, -10, 10)
                
                # f(1) is determined by sum f_k/k = 0
                S = np.sum(f_vals / ks)
                f1 = np.clip(-S, -10, 10)
                
                domain = np.concatenate(([1], ks))
                values = np.concatenate(([f1], f_vals))
                
                # Verify M(x) <= 1.0001 for all x in [1, 10K]
                K = int(ks.max())
                max_x_check = 10 * K
                feasible = True
                for x in range(1, max_x_check + 1):
                    Mx = f1 * x
                    for j, k in enumerate(ks):
                        Mx += f_vals[j] * (x // k)
                    if Mx > 1.0001:
                        feasible = False
                        break
                
                if feasible:
                    return domain, values
        except Exception:
            pass
        return None
    
    # Try increasing K_max values, starting from smaller for speed
    K_candidates = [50, 100, 200, 300, 500]
    
    for K_max in K_candidates:
        if time.time() > deadline:
            break
            
        result = solve_lp_for_K(K_max)
        if result is not None:
            return result
    
    # Fallback to Mobius
    K = 200000
    mu = _compute_mobius(K)
    mask = mu[1:] != 0
    domain = np.where(mask)[0] + 1
    values = mu[domain].astype(np.float64)
    return domain, values
