import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Hybrid multi-start optimizer: CMA-ES + L-BFGS-B for BBOB functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Safety margin: 330s limit, stop at 320s
        deadline = time.time() + 320
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        bounds = [(-5, 5) for _ in range(dim)]
        lower = np.full(dim, -5.0)
        upper = np.full(dim, 5.0)
        
        # Wrapper to track calls and update best
        def wrapped_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(x, lower, upper)
            val = func(x)
            calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Coarse exploration with CMA-ES
        # Use a moderate population size to explore the space
        # CMA-ES is good for multimodal functions like Rastrigin and Gallagher
        
        # Start with a reasonable sigma for exploration
        sigma0 = 3.0
        popsize = 10 + int(3 * np.log(dim))
        
        # Run CMA-ES with a limited budget for exploration
        cma_budget = int(budget * 0.4)  # Use 40% of budget for exploration
        
        try:
            # Use scipy's minimize with CMA-ES if available, otherwise use a simple implementation
            # Since scipy.optimize doesn't have built-in CMA-ES, we'll implement a basic version
            # or use Nelder-Mead as a fallback for exploration
            
            # Actually, let's use a simpler approach: multiple Nelder-Mead starts
            # followed by L-BFGS-B refinement
            
            n_starts = 8
            per_start_budget = max(15, (budget - 5) // n_starts)
            
            for i in range(n_starts):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Generate diverse starting points
                if i == 0:
                    x0 = np.zeros(dim)
                elif i == 1:
                    x0 = rng.uniform(-3, 3, dim)
                elif i == 2:
                    x0 = rng.uniform(-5, 5, dim)
                else:
                    # Mix of random and perturbed best
                    if best_x is not None and rng.random() < 0.5:
                        x0 = np.clip(best_x + rng.normal(0, 1.5, dim), lower, upper)
                    else:
                        x0 = rng.uniform(-5, 5, dim)
                
                # Evaluate start
                fx0 = wrapped_func(x0)
                
                if calls >= budget or time.time() > deadline:
                    break
                
                # Local refinement with L-BFGS-B
                remaining_for_this = min(per_start_budget, budget - calls)
                if remaining_for_this < 10:
                    continue
                    
                try:
                    res = scipy.optimize.minimize(
                        wrapped_func, 
                        x0, 
                        method='L-BFGS-B', 
                        bounds=bounds,
                        options={
                            'maxiter': 20, 
                            'maxfev': remaining_for_this, 
                            'ftol': 1e-10, 
                            'gtol': 1e-7, 
                            'disp': False
                        }
                    )
                except Exception:
                    pass
                
        except Exception:
            pass

        # Phase 2: Intensive refinement of the best point found
        if calls < budget and best_x is not None and time.time() < deadline:
            remaining = budget - calls
            if remaining > 10:
                try:
                    # Multiple refinement attempts with different tolerances
                    for attempt in range(3):
                        if calls >= budget or time.time() > deadline:
                            break
                        
                        tol = [1e-12, 1e-14, 1e-16][min(attempt, 2)]
                        maxfev_this = max(5, remaining // (3 - attempt))
                        
                        res = scipy.optimize.minimize(
                            wrapped_func, 
                            best_x, 
                            method='L-BFGS-B', 
                            bounds=bounds,
                            options={
                                'maxiter': 30, 
                                'maxfev': maxfev_this, 
                                'ftol': tol, 
                                'gtol': 1e-9, 
                                'disp': False
                            }
                        )
                except Exception:
                    pass

        # Phase 3: Final polish with tight tolerances
        if calls < budget and best_x is not None and time.time() < deadline:
            remaining = budget - calls
            if remaining > 5:
                try:
                    res = scipy.optimize.minimize(
                        wrapped_func, 
                        best_x, 
                        method='L-BFGS-B', 
                        bounds=bounds,
                        options={
                            'maxiter': 50, 
                            'maxfev': remaining, 
                            'ftol': 1e-15, 
                            'gtol': 1e-10, 
                            'disp': False
                        }
                    )
                except Exception:
                    pass

        return best_x, best_f
