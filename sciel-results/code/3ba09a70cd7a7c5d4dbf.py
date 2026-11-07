"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 285s margin before 300s limit
        rng = np.random.default_rng(42)
        
        # Track best solution
        best_x = None
        best_f = float('inf')
        calls = 0
        
        def tracked_func(x):
            nonlocal calls
            if calls >= self.budget:
                return 0.0
            calls += 1
            return func(x)
        
        # Phase 1: Try L-BFGS-B from multiple starting points
        # Start with origin and a few random points
        start_points = [np.zeros(self.dim)]
        for _ in range(10):
            start_points.append(rng.uniform(-4, 4, self.dim))
        
        for sp in start_points:
            if time.time() >= deadline:
                break
            if calls >= self.budget:
                break
            
            try:
                res = minimize(tracked_func, sp, method='L-BFGS-B',
                             bounds=[(-5, 5)] * self.dim,
                             options={'maxiter': 100, 'maxfun': 50})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        # Phase 2: If we have budget left, try a few more random starts
        # focusing on areas near the best solution found
        if best_x is not None and calls < self.budget and time.time() < deadline:
            for _ in range(5):
                if time.time() >= deadline:
                    break
                if calls >= self.budget:
                    break
                
                # Perturb around best solution
                perturbation = rng.normal(0, 1.0, self.dim)
                sp = np.clip(best_x + perturbation, -5, 5)
                
                try:
                    res = minimize(tracked_func, sp, method='L-BFGS-B',
                                 bounds=[(-5, 5)] * self.dim,
                                 options={'maxiter': 50, 'maxfun': 20})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass
        
        # Phase 3: Use remaining budget for a fine-tuning (1+1)-ES from best solution
        if best_x is not None and calls < self.budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            
            while calls < self.budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = tracked_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.85
        
        # Ensure we return a valid point
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = tracked_func(best_x)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
