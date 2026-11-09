import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s margin before 300s limit
        
        n = self.dim
        budget = self.budget
        
        # Global state
        best_x = None
        best_f = np.inf
        eval_count = 0
        rng = np.random.default_rng(42)

        def safe_func(x):
            nonlocal best_x, best_f, eval_count
            if time.time() > deadline:
                raise TimeoutError
            if eval_count >= budget:
                raise BudgetExceeded
            f_val = func(x)
            eval_count += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def nelder_mead(x0, initial_step=0.5, max_iter=50):
            nonlocal best_x, best_f, eval_count
            
            # Initialize simplex
            simplex = [x0.copy()]
            for i in range(n):
                x_i = x0.copy()
                x_i[i] += initial_step
                simplex.append(x_i)
            
            f_vals = [safe_func(s) for s in simplex]
            
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink
            
            for _ in range(max_iter):
                if time.time() > deadline or eval_count >= budget:
                    break
                
                # Sort simplex
                order = np.argsort(f_vals)
                simplex = [simplex[i] for i in order]
                f_vals = [f_vals[i] for i in order]
                
                # Check convergence (simple heuristic)
                if f_vals[-1] - f_vals[0] < 1e-6:
                    break
                    
                # Centroid of all but worst point
                x_centroid = np.mean(simplex[:-1], axis=0)
                
                # 1. Reflection
                x_r = x_centroid + alpha * (x_centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = safe_func(x_r)
                
                if f_r < f_vals[0]:
                    # 2. Expansion
                    x_e = x_centroid + gamma * (x_r - x_centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = safe_func(x_e)
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_vals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                elif f_r < f_vals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
                else:
                    # 3. Contraction
                    x_c = x_centroid + rho * (simplex[-1] - x_centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = safe_func(x_c)
                    
                    if f_c < f_vals[-1]:
                        simplex[-1] = x_c
                        f_vals[-1] = f_c
                    else:
                        # 4. Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_vals[i] = safe_func(simplex[i])
                            
                if eval_count >= budget:
                    break

            return simplex[0], f_vals[0]

        # Phase 1: Global Sampling
        n_samples = 20
        for i in range(n_samples):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, n)
            safe_func(x)
            
        if best_x is None:
            return np.zeros(n), 0.0

        # Phase 2: Multi-Start Nelder-Mead
        # We use the best found so far and some other good points as starts
        # Sort all evaluated points? We don't store them all easily without extra bookkeeping.
        # Let's just use the current best and generate a few more random starts if budget allows.
        
        # Run NM from the best point found in Phase 1
        x_start = best_x.copy()
        x_start, _ = nelder_mead(x_start, initial_step=1.0, max_iter=50)
        
        # If we have budget left, try restarting from other random points
        # The Lunacek function has a secondary minimum. 
        # Heuristic: Try a few random restarts if we haven't converged to a very low value.
        # But we don't know f_opt. 
        # Let's just run NM again from a different random point if we have > 50 evals left.
        
        if eval_count < budget - 20:
            x_new = rng.uniform(-4, 4, n)
            x_new, _ = nelder_mead(x_new, initial_step=1.0, max_iter=50)
            
        if eval_count < budget - 20:
            x_new = rng.uniform(-4, 4, n)
            x_new, _ = nelder_mead(x_new, initial_step=1.0, max_iter=50)

        return best_x, best_f

    def _cov_to_b_d(self, C, n):
        # Not used in NM strategy, but kept for interface compatibility if needed
        try:
            L = np.linalg.cholesky(C)
            return L, np.ones(n)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            return eigvecs, np.sqrt(eigvals)
