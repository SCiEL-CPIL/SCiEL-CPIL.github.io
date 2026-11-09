"""Directional search exploiting Bent Cigar anisotropy: isolate dominant axis, optimize 1D, then polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Directional search optimizer for anisotropic functions like Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = int(self.budget)
        t0 = time.time()
        deadline = t0 + 280.0
        max_calls = budget

        calls = [0]
        best_fx = float('inf')
        best_x = np.zeros(dim)
        rng = np.random.default_rng(42)

        def f(x):
            nonlocal best_fx, best_x
            if calls[0] >= max_calls:
                return float('inf')
            calls[0] += 1
            try:
                x_clipped = np.clip(x, -5.0, 5.0)
                val = float(func(x_clipped))
            except Exception:
                return float('inf')
            if not np.isfinite(val):
                return float('inf')
            if val < best_fx:
                best_fx = val
                best_x = x_clipped.copy()
            return val

        def remaining():
            return deadline - time.time()

        # Phase 1: Global Exploration via Random Sampling
        # Spend 40% of budget on random sampling to find good starting points
        n_random = int(budget * 0.4)
        if n_random < 10:
            n_random = 10
        
        candidates = []
        for _ in range(n_random):
            if calls[0] >= max_calls or remaining() < 5.0:
                break
            x_rand = rng.uniform(-5.0, 5.0, dim)
            fx_rand = f(x_rand)
            candidates.append((fx_rand, x_rand))
        
        if not candidates:
            # Fallback if no calls were made
            if calls[0] < max_calls:
                x0 = np.zeros(dim)
                f(x0)
            return best_x, best_fx

        # Sort candidates by function value
        candidates.sort(key=lambda x: x[0])
        
        # Phase 2: Local Refinement using SciPy Powell's Method
        # Try the best 3 candidates
        n_local = min(3, len(candidates))
        
        for i in range(n_local):
            if calls[0] >= max_calls or remaining() < 10.0:
                break
            
            start_x = candidates[i][1]
            
            # Define a wrapper for scipy that checks budget
            def scipy_func(x):
                if calls[0] >= max_calls or remaining() < 1.0:
                    return float('inf')
                return f(x)
            
            # Use Powell's method which is good for anisotropic functions
            try:
                from scipy.optimize import minimize
                res = minimize(scipy_func, start_x, method='Powell',
                               options={'maxiter': 50, 'maxfev': max_calls - calls[0], 'disp': False})
                if res.success and res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x
            except Exception:
                pass
            
            # If budget is low, break early
            if calls[0] >= max_calls * 0.8:
                break

        # Phase 3: Final Polish if budget remains
        if calls[0] < max_calls and remaining() > 5.0:
             # Run another round of Powell from the current best
             def scipy_func_final(x):
                if calls[0] >= max_calls or remaining() < 1.0:
                    return float('inf')
                return f(x)
             
             try:
                from scipy.optimize import minimize
                res = minimize(scipy_func_final, best_x, method='Powell',
                               options={'maxiter': 20, 'maxfev': max_calls - calls[0], 'disp': False})
                if res.success and res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x
             except Exception:
                pass

        if best_x is None or not np.all(np.isfinite(best_x)):
            best_x = np.zeros(dim)
            if calls[0] < max_calls:
                best_fx = f(best_x)

        return best_x, best_fx
# EVOLVE-BLOCK-END
