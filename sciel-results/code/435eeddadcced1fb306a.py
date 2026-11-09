import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 310
        
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        eval_count = 0
        dim = self.dim
        
        # Helper: Evaluate and update best
        def evaluate(x):
            nonlocal best_x, best_f, eval_count
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx
        
        # Phase 1: Initial diverse sampling to find promising basins
        n_init = 40
        rng = np.random.default_rng(42)
        
        for i in range(n_init):
            if eval_count >= self.budget or time.time() > deadline - 10:
                break
            x = rng.uniform(-5, 5, size=dim)
            evaluate(x)
        
        # Phase 2: Multi-Start Nelder-Mead
        # Sort initial points by fitness and run Nelder-Mead from top candidates
        # We'll keep track of all evaluated points to select good starting points
        
        # Collect all evaluated points from Phase 1
        # We need to store them, so let's modify evaluate to also track points
        # Actually, let's just use the best point and some random restarts
        
        # Run Nelder-Mead from multiple starting points
        n_nm_starts = 5
        nm_budget_per_start = (self.budget - eval_count) // (n_nm_starts + 1)
        
        # Use best point and some other good points as starts
        start_points = [best_x.copy()]
        
        # Add a few random points near the best as additional starts
        for i in range(n_nm_starts - 1):
            if eval_count >= self.budget or time.time() > deadline - 10:
                break
            # Perturb the best point
            x_start = np.clip(best_x + 1.0 * rng.standard_normal(dim), -5, 5)
            start_points.append(x_start)
        
        for start_idx, x_start in enumerate(start_points):
            if eval_count >= self.budget - 10 or time.time() > deadline - 10:
                break
            
            # Run Nelder-Mead from this starting point
            # Use a callback to track evaluations
            nm_eval_count = [0]
            
            def nm_func(x):
                nm_eval_count[0] += 1
                return evaluate(x)
            
            try:
                result = minimize(
                    nm_func,
                    x_start,
                    method='Nelder-Mead',
                    options={
                        'maxiter': nm_budget_per_start,
                        'adaptive': True,
                        'xatol': 1e-6,
                        'fatol': 1e-8
                    }
                )
            except Exception:
                pass
            
            # Update best if Nelder-Mead found something better
            # (evaluate already updates best_x and best_f)
        
        # Phase 3: Final local polish with (1+1)-ES
        remaining = self.budget - eval_count
        if remaining > 0 and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            no_improve = 0
            limit = 30
            
            while remaining > 0 and time.time() < deadline - 5:
                if no_improve > limit:
                    # Restart with larger sigma
                    sigma = 0.1
                    no_improve = 0
                    if sigma > 1.0:
                        break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                remaining -= 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                    no_improve = 0
                else:
                    sigma *= 0.9
                    no_improve += 1
                    if sigma < 1e-5:
                        break
        
        return best_x, best_f
