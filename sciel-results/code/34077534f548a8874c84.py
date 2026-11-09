import numpy as np
import time
import math
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 320.0
        self.dim = 5
        self.budget = 500

        best_x = np.zeros(self.dim)
        best_f = float('inf')
        eval_count = 0

        def check_deadline():
            return time.time() < deadline - 3.0

        def safe_eval(x):
            nonlocal eval_count, best_f, best_x
            if eval_count >= self.budget:
                return None
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = func(x_clipped)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Initial sampling to find good starting points
        n_init = 50
        for i in range(n_init):
            if not check_deadline() or eval_count >= self.budget:
                break
            x = rng.uniform(-5.0, 5.0, self.dim)
            fx = safe_eval(x)
            if fx is None:
                break

        if eval_count >= self.budget or not check_deadline():
            return best_x, best_f

        # Phase 2: Multi-start Nelder-Mead
        # Sort initial points by fitness and use top ones as starts
        # We'll do a few NM runs from different starting points
        
        # Collect some good starting points
        start_points = []
        # Add best point
        start_points.append(best_x.copy())
        
        # Add a few random perturbations of best point
        for _ in range(3):
            if eval_count >= self.budget or not check_deadline():
                break
            x_try = best_x + rng.normal(0, 0.5, self.dim)
            fx = safe_eval(x_try)
            if fx is None:
                break
            if fx < best_f + 1.0:  # Only keep if reasonably good
                start_points.append(x_try.copy())
        
        # Also try a few independent random points
        for _ in range(5):
            if eval_count >= self.budget or not check_deadline():
                break
            x_try = rng.uniform(-5.0, 5.0, self.dim)
            fx = safe_eval(x_try)
            if fx is None:
                break
            if fx < best_f + 5.0:
                start_points.append(x_try.copy())

        # Run Nelder-Mead from each start point
        for x0 in start_points:
            if eval_count >= self.budget or not check_deadline():
                break
            
            def nm_func(x):
                nonlocal eval_count
                if eval_count >= self.budget:
                    return best_f
                x_clipped = np.clip(x, -5.0, 5.0)
                fx = func(x_clipped)
                eval_count += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x_clipped.copy()
                return fx
            
            try:
                res = minimize(nm_func, x0, method='Nelder-Mead',
                              options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True})
            except Exception:
                pass

            if not check_deadline():
                break

        # Phase 3: Final polish with (1+1)-ES
        if eval_count < self.budget and check_deadline():
            x_cur = best_x.copy()
            f_cur = best_f
            sigma = 0.1
            no_improve = 0
            stall_limit = 15
            
            while eval_count < self.budget and check_deadline():
                y = np.clip(x_cur + sigma * rng.standard_normal(self.dim), -5.0, 5.0)
                fy = func(y)
                eval_count += 1
                
                if fy < f_cur:
                    x_cur = y
                    f_cur = fy
                    if f_cur < best_f:
                        best_f = f_cur
                        best_x = x_cur.copy()
                    no_improve = 0
                    sigma *= 1.03
                else:
                    no_improve += 1
                    sigma *= 0.97
                    
                    if no_improve >= stall_limit:
                        break
                
                if sigma < 1e-5:
                    break

        return best_x, best_f
