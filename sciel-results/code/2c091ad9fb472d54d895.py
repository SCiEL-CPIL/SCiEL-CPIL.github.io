import numpy as np
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        total_calls = 0
        best_x = None
        best_f = np.inf
        
        # We will run multiple Nelder-Mead optimizations.
        # Each run has a max budget.
        # If we find a good solution, we stop.
        
        max_runs = 20  # Max number of restarts
        budget_per_run = 30  # Budget for each Nelder-Mead run
        
        for i in range(max_runs):
            if total_calls >= self.budget:
                break
            
            # Initial point
            # Use a mix of random and slightly perturbed best
            if i == 0:
                x0 = rng.uniform(-5, 5, self.dim)
            else:
                # 50% chance to restart randomly, 50% to perturb best
                if rng.random() < 0.5:
                    x0 = rng.uniform(-5, 5, self.dim)
                else:
                    x0 = np.clip(best_x + rng.normal(0, 1.0, self.dim), -5, 5)
            
            # Wrapper to count calls
            calls = 0
            def wrapper(x):
                nonlocal calls
                calls += 1
                return func(x)
            
            # Determine maxfev for this run
            remaining = self.budget - total_calls
            maxfev = min(budget_per_run, remaining)
            
            # Run Nelder-Mead
            try:
                res = scipy.optimize.minimize(
                    wrapper, x0, method='Nelder-Mead',
                    options={
                        'maxiter': maxfev,
                        'maxfev': maxfev,
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'adaptive': True
                    }
                )
            except Exception:
                # If minimize fails, try to get the best point from the history?
                # Or just skip.
                pass
            
            total_calls += calls
            
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
            
            # If we found a very good solution, stop
            # Rastrigin global min is ~0. Local ~50.
            # If best_f < 5, we are likely in the global basin.
            if best_f < 5:
                break
        
        return best_x, best_f
