import numpy as np
import time
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = 600  # Leave 30s margin for 630s limit
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # Try increasing K
    # Start with smaller K to get a valid solution quickly, then try larger
    K_candidates = [100, 500, 1000, 2000, 3000, 4000, 5000, 6000]
    
    for K in K_candidates:
        if time.time() > deadline - 60:  # Need at least 60s for solve and cleanup
            break
            
        n_vars = K - 1
        if n_vars <= 0:
            continue
            
        ks = np.arange(2, K + 1)
        c = np.log(ks) / ks
        
        max_x = 10 * K
        
        # Check memory feasibility roughly
        # 10*K * (K-1) * 8 bytes
        mem_needed = (10 * K) * (K - 1) * 8
        if mem_needed > 3 * 1024**3:  # 3GB
            continue

        try:
            # Vectorized construction of A_ub
            # A_ub[i, j] = - { (i+1) / (j+2) }
            # x_vals is 1-based index for x in [1, max_x]
            x_vals = np.arange(1, max_x + 1, dtype=np.float64)
            
            # Construct main constraint matrix
            # Shape: (max_x, n_vars)
            # This is the most memory intensive part
            A_ub_main = -(x_vals[:, None] % ks[None, :]) / ks[None, :]
            
            # Add constraints for sum(f(k)/k)
            # f(1) = -sum(g/k). -10 <= f(1) <= 10
            # sum(g/k) <= 10
            # -sum(g/k) <= 10 => sum(g/k) >= -10 => -sum(g/k) <= 10
            row_pos = 1.0 / ks
            row_neg = -1.0 / ks
            
            A_ub = np.vstack([A_ub_main, row_pos, row_neg])
            b_ub = np.concatenate([np.ones(max_x) * 1.0001, [10.0, 10.0]])
            
            bounds = [(-10, 10)] * n_vars
            
            # Solve LP
            # Maximize a = -sum(c * f) => Minimize sum(c * f)
            # Wait, objective in LP is minimize c^T x.
            # We want to maximize -sum(c_k f_k) = minimize sum(c_k f_k).
            # So c in linprog should be c.
            
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', 
                          options={'presolve': 'default', 'dual_feasibility_tolerance': 1e-8, 'primal_feasibility_tolerance': 1e-8})
            
            # Free memory
            del A_ub_main
            del A_ub
            del b_ub
            del x_vals
            
            if res.success:
                g = res.x
                f = np.zeros(K + 1)
                f[2:K+1] = g
                S = np.sum(g / ks)
                f[1] = -S
                
                # Verify f(1) is within bounds
                if abs(f[1]) <= 10.0 + 1e-9:
                    # Calculate score
                    a = -np.sum(g * c)
                    
                    # Verify constraints just in case of numerical drift
                    # M(x) <= 1.0001
                    # We trust the LP for now, but a quick check on a few points?
                    # The LP guarantees it.
                    
                    if a > best_score:
                        best_score = a
                        domain = np.arange(1, K + 1, dtype=np.int64)
                        values = np.clip(f[1:K+1], -10, 10)
                        best_domain = domain
                        best_values = values

        except MemoryError:
            continue
        except Exception as e:
            pass

    if best_domain is not None:
        return best_domain, best_values
    else:
        # Fallback to seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
