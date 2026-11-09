import numpy as np
import time
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 550  # Safety margin
    
    # Seed solution
    best_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    best_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    
    # Calculate seed score to have a baseline
    # f(1) is adjusted.
    # Seed: f(1)=3.47, f(2)=-1, f(3)=-1, f(5)=-1, f(30)=1
    # S = sum f(k)/k = 3.47/1 - 1/2 - 1/3 - 1/5 + 1/30
    # S = 3.47 - 0.5 - 0.333 - 0.2 + 0.033 = 2.466
    # f(1)_new = 3.47 - 2.466 = 1.004
    # a = - (1.004*0 - 1*ln2/2 - 1*ln3/3 - 1*ln5/5 + 1*ln30/30)
    # a = - ( -0.346 - 0.366 - 0.321 + 0.098 ) = - ( -0.923 ) = 0.923 ?
    # Wait, the seed score is 1.0.
    # Let's trust the evaluator's score of 1.0.
    
    best_score = 1.0
    
    # List of K to try
    K_list = [100, 200, 500, 1000]
    
    for K in K_list:
        if time.time() > deadline:
            break
            
        k = np.arange(2, K + 1)
        c = np.log(k) / k
        
        x = np.arange(1, 10 * K + 1)
        
        # Check memory/time feasibility
        # For K=1000, A is 10000x999.
        
        try:
            # Construct A_ub
            # Using broadcasting
            mods = x[:, None] % k[None, :]
            A_ub = -mods / k[None, :]
            
            b_ub = np.full(len(x), 1.0001, dtype=np.float64)
            
            bounds = [(-10.0, 10.0)] * len(k)
            
            # Use HiGHS
            # Time limit for solver
            solver_time_limit = 100
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options={'time_limit': solver_time_limit})
            
            if res.status == 0:
                f = res.x
                mask = np.abs(f) > 1e-6
                if not np.any(mask):
                    continue
                
                domain_k = k[mask]
                values_f = f[mask]
                
                # Calculate score
                score = -np.sum(values_f * np.log(domain_k) / domain_k)
                
                # Optional: Verify constraints
                # M(x) = - sum f(k) {x/k}
                # We used -A_ub in solver?
                # Solver: A_ub f <= b_ub
                # A_ub = - {x/k}
                # - {x/k} f <= 1.0001
                # sum f(k) {x/k} >= -1.0001
                # This matches M(x) <= 1.0001 derived earlier.
                
                if score > best_score:
                    best_score = score
                    best_domain = domain_k
                    best_values = values_f
                    
        except Exception:
            continue
            
    return best_domain, best_values
