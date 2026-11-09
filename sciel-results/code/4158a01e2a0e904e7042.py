import numpy as np
import time

class Optimizer:
    """
    Multi-restart (1+8)-ES with per-dimension adaptive step sizes
    for BBOB f18 (Schaffers F7 ill-conditioned).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0

        calls = 0
        best_x = None
        best_f = np.inf

        def eval_pt(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Sobol quasi-random global exploration
        from scipy.stats import qmc
        n_init = min(10, budget)
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        init_points = sampler.random(n=n_init)
        init_points = 2.0 * init_points - 1.0
        init_points = 5.0 * init_points
        
        for i in range(n_init):
            if calls >= budget or time.time() > deadline:
                break
            eval_pt(init_points[i])
            
        if best_x is None:
            best_x = np.zeros(dim)
            eval_pt(best_x)

        # Phase 2: Multi-restart (1+8)-ES with per-dimension adaptive step sizes
        # This approach uses multiple restarts to escape local minima,
        # which is crucial for the ill-conditioned Schaffers F7 landscape.
        n_restarts = 5
        lam = 8  # population size for (1+8)-ES
        
        for restart in range(n_restarts):
            if calls >= budget or time.time() > deadline:
                break
            
            # For the first restart, start from the best found so far
            # For subsequent restarts, perturb from the best point
            if restart == 0:
                mean = best_x.copy()
            else:
                # Perturb from best with increasing radius
                perturb_radius = 1.0 * (restart + 1)
                mean = best_x + perturb_radius * np.random.randn(dim)
                mean = np.clip(mean, -5.0, 5.0)
                eval_pt(mean)
            
            # Per-dimension step sizes, initialized based on distance from center
            sigma = np.abs(mean) * 0.5 + 0.5
            sigma = np.clip(sigma, 0.1, 5.0)
            
            # Adaptive step size parameters (1/5 rule)
            success_rate = 0.5
            n_success = 0
            n_fail = 0
            window_size = 20
            
            max_iter = (budget - calls) // lam + 1
            for it in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Generate offspring
                offspring = []
                for i in range(lam):
                    if calls >= budget or time.time() > deadline:
                        break
                    z = np.random.randn(dim)
                    x_child = mean + sigma * z
                    x_child = np.clip(x_child, -5.0, 5.0)
                    offspring.append(x_child)
                
                if len(offspring) < 1:
                    break
                
                # Evaluate offspring
                fvals = [eval_pt(x) for x in offspring]
                
                # Find best offspring
                best_idx = np.argmin(fvals)
                best_child = offspring[best_idx]
                best_child_f = fvals[best_idx]
                
                # Update mean and step sizes using 1/5 rule
                if best_child_f < eval_pt(mean):
                    # Success: increase step sizes
                    mean = best_child.copy()
                    sigma = np.minimum(sigma * 1.05, 5.0)
                    n_success += 1
                else:
                    # Failure: decrease step sizes
                    sigma = np.maximum(sigma * 0.95, 0.01)
                    n_fail += 1
                
                # Adapt step sizes based on success rate over window
                if n_success + n_fail >= window_size:
                    actual_rate = n_success / (n_success + n_fail)
                    if actual_rate > success_rate + 0.1:
                        sigma = np.minimum(sigma * 1.1, 5.0)
                    elif actual_rate < success_rate - 0.1:
                        sigma = np.maximum(sigma * 0.9, 0.01)
                    n_success = 0
                    n_fail = 0
                
                # Check for convergence
                if np.max(sigma) - np.min(sigma) < 1e-6 and np.all(sigma < 0.01):
                    break
            
            # After each restart, update global best
            if best_x is not None:
                eval_pt(best_x)
        
        # Phase 3: Final local refinement with Nelder-Mead if budget remains
        if calls < budget and best_x is not None and time.time() < deadline:
            from scipy.optimize import minimize
            x0 = best_x.copy()
            options = {'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10}
            try:
                res = minimize(func, x0, method='Nelder-Mead', options=options)
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass
        
        return best_x, best_f
