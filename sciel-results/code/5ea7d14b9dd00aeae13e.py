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
        best_x = np.zeros(n)
        best_f = np.inf
        eval_count = 0
        
        class _BudgetExceeded(Exception):
            pass
        
        def safe_func(x):
            nonlocal best_x, best_f, eval_count
            if time.time() > deadline:
                raise _BudgetExceeded
            if eval_count >= budget:
                raise _BudgetExceeded
            f_val = func(np.array(x, dtype=np.float64))
            eval_count += 1
            if f_val < best_f:
                best_f = f_val
                best_x = np.array(x, dtype=np.float64)
            return f_val

        # Phase 1: Initial sampling to find a good starting point
        n_samples = min(30, budget)
        rng = np.random.default_rng(42)
        
        # First, try origin
        f_origin = safe_func(np.zeros(n))
        
        # Then random samples
        for i in range(1, n_samples):
            x = rng.uniform(-5, 5, n)
            safe_func(x)
            
            if eval_count >= budget:
                break

        if best_x is None:
            return np.zeros(n), 0.0

        # Phase 2: Nelder-Mead with adaptive vertex shrinking and periodic re-seeding
        # This exploits local curvature more effectively than population-based methods
        # and handles the bi-modal nature of Lunacek well
        
        def nelder_mead(x0, max_iter=100):
            nonlocal best_x, best_f, eval_count
            nonlocal deadline, budget
            
            # Initialize simplex
            simplex = [x0.copy()]
            f0 = safe_func(x0)
            simplex_f = [f0]
            
            # Add vertices
            for i in range(n):
                v = x0.copy()
                step = 0.25 if abs(x0[i]) > 0.1 else 0.25
                v[i] += step
                v = np.clip(v, -5, 5)
                simplex.append(v)
                simplex_f.append(safe_func(v))
                
                if eval_count >= budget or time.time() > deadline:
                    return
            
            # Nelder-Mead parameters
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma_shrink = 0.5  # shrinkage
            
            for iteration in range(max_iter):
                if eval_count >= budget or time.time() > deadline:
                    return
                
                # Sort simplex by fitness
                idx = np.argsort(simplex_f)
                simplex = [simplex[i] for i in idx]
                simplex_f = [simplex_f[i] for i in idx]
                
                # Check convergence
                if simplex_f[0] < best_f:
                    best_f = simplex_f[0]
                    best_x = simplex[0].copy()
                
                # If the spread is too small, we've converged
                centroid = np.mean(simplex[1:], axis=0)
                spread = np.max([np.linalg.norm(v - centroid) for v in simplex])
                if spread < 1e-8:
                    break
                
                # Reflection
                x_reflect = centroid + alpha * (centroid - simplex[-1])
                x_reflect = np.clip(x_reflect, -5, 5)
                f_reflect = safe_func(x_reflect)
                
                if eval_count >= budget or time.time() > deadline:
                    return
                
                if f_reflect < simplex_f[0]:
                    # Expansion
                    x_expand = centroid + gamma * (x_reflect - centroid)
                    x_expand = np.clip(x_expand, -5, 5)
                    f_expand = safe_func(x_expand)
                    
                    if eval_count >= budget or time.time() > deadline:
                        return
                    
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        simplex_f[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        simplex_f[-1] = f_reflect
                elif f_reflect < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = x_reflect
                    simplex_f[-1] = f_reflect
                else:
                    # Contraction
                    if f_reflect < simplex_f[-1]:
                        # Outside contraction
                        x_contract = centroid + rho * (x_reflect - centroid)
                    else:
                        # Inside contraction
                        x_contract = centroid - rho * (x_reflect - centroid)
                    
                    x_contract = np.clip(x_contract, -5, 5)
                    f_contract = safe_func(x_contract)
                    
                    if eval_count >= budget or time.time() > deadline:
                        return
                    
                    if f_contract <= min(f_reflect, simplex_f[-1]):
                        simplex[-1] = x_contract
                        simplex_f[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = safe_func(simplex[i])
                            
                            if eval_count >= budget or time.time() > deadline:
                                return
                
                # Update best
                if simplex_f[0] < best_f:
                    best_f = simplex_f[0]
                    best_x = simplex[0].copy()
            
            return

        # Run Nelder-Mead from the best initial point
        nelder_mead(best_x.copy(), max_iter=150)
        
        # Phase 3: Periodic re-seeding to explore other basins
        # Lunacek has two global minima; we need to explore both
        remaining_budget = budget - eval_count
        if remaining_budget > 20 and time.time() < deadline:
            # Try a few different starting points to escape local minima
            n_restarts = min(5, remaining_budget // 40)
            for r in range(n_restarts):
                if eval_count >= budget or time.time() > deadline:
                    break
                
                # Generate a new starting point: mix of random and perturbed best
                if r % 2 == 0:
                    x_start = best_x + rng.normal(0, 0.5, n)
                else:
                    x_start = rng.uniform(-5, 5, n)
                
                x_start = np.clip(x_start, -5, 5)
                
                # Run a shorter Nelder-Mead
                nelder_mead(x_start, max_iter=30)
        
        # Phase 4: Final local refinement around the best point
        if eval_count < budget and time.time() < deadline:
            # Do a fine-grained search around the best point
            for _ in range(min(20, budget - eval_count)):
                if time.time() > deadline:
                    break
                x = best_x + rng.normal(0, 0.05, n)
                x = np.clip(x, -5, 5)
                safe_func(x)
        
        return best_x, best_f
