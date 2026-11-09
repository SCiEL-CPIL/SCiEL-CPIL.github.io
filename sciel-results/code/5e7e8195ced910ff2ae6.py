import numpy as np
import time

class Optimizer:
        def __init__(self, budget, dim, n_obj):
            self.budget = budget
            self.dim = dim
            self.n_obj = n_obj

        def __call__(self, problem, bounds):
            start_time = time.time()
            deadline = start_time + 290.0
            
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            
            n_bias = self.dim - 3
            conv_idx = slice(n_bias, self.dim)
            bias_idx = slice(0, n_bias)
            
            # 1. Calibration: Find good convergence variables
            # Use a central biasing point
            x_bias_center = (lo[bias_idx] + hi[bias_idx]) / 2.0
            
            best_conv = None
            best_val = np.inf
            # 10 random samples for convergence
            for _ in range(10):
                if time.time() > deadline: break
                x_c = lo[conv_idx] + (hi[conv_idx] - lo[conv_idx]) * np.random.random(3)
                x = np.empty(self.dim)
                x[bias_idx] = x_bias_center
                x[conv_idx] = x_c
                f = problem(x)
                val = np.sum(f)
                if val < best_val:
                    best_val = val
                    best_conv = x_c
            
            if best_conv is None:
                best_conv = (lo[conv_idx] + hi[conv_idx]) / 2.0
                
            # 2. Exploration: 150 points
            n_explore = 150
            X_exp = np.empty((n_explore, self.dim))
            F_exp = np.empty((n_explore, self.n_obj))
            
            for i in range(n_explore):
                if time.time() > deadline:
                    X_exp = X_exp[:i]
                    F_exp = F_exp[:i]
                    break
                x = np.empty(self.dim)
                x[bias_idx] = lo[bias_idx] + (hi[bias_idx] - lo[bias_idx]) * np.random.random(n_bias)
                x[conv_idx] = best_conv
                X_exp[i] = x
                F_exp[i] = problem(x)
            
            # 3. Refinement: 40 points
            # Find non-dominated points in F_exp
            # Simple non-dominated check
            n_points = len(F_exp)
            is_dom = np.zeros(n_points, dtype=bool)
            for i in range(n_points):
                for j in range(n_points):
                    if i == j: continue
                    if np.all(F_exp[j] <= F_exp[i]) and np.any(F_exp[j] < F_exp[i]):
                        is_dom[i] = True
                        break
            
            non_dom_idx = np.where(~is_dom)[0]
            
            # Pick up to 20 non-dominated points to refine
            if len(non_dom_idx) > 0:
                # Shuffle to get diverse ones? Or just take first 20.
                # Taking first 20 is fine.
                refine_idx = non_dom_idx[:20]
                n_refine = 40
                X_ref = np.empty((n_refine, self.dim))
                F_ref = np.empty((n_refine, self.n_obj))
                
                for k in range(n_refine):
                    if time.time() > deadline:
                        X_ref = X_ref[:k]
                        F_ref = F_ref[:k]
                        break
                    # Pick a parent
                    parent_idx = refine_idx[k % len(refine_idx)]
                    parent_x = X_exp[parent_idx]
                    
                    # Perturb biasing variables
                    # Small step
                    step = 0.05 * (hi[bias_idx] - lo[bias_idx])
                    noise = np.random.normal(0, 1, n_bias) * step
                    new_bias = np.clip(parent_x[bias_idx] + noise, lo[bias_idx], hi[bias_idx])
                    
                    x = np.empty(self.dim)
                    x[bias_idx] = new_bias
                    x[conv_idx] = best_conv # Keep convergence fixed
                    
                    X_ref[k] = x
                    F_ref[k] = problem(x)
                    
                # Combine results
                X_final = np.vstack([X_exp, X_ref])
                F_final = np.vstack([F_exp, F_ref])
            else:
                X_final = X_exp
                F_final = F_exp
                
            return X_final
