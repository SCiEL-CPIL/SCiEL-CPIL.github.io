import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        n = self.dim
        budget = self.budget
        deadline = start_time + 280

        best_x = np.zeros(n)
        best_f = float('inf')
        calls = 0

        # Wrapper to track calls
        def wrapper(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                # Return a large value to signal failure/stop? 
                # Or raise an exception?
                # Raising an exception in minimize might be messy.
                # Returning a large value might cause it to keep going if it thinks it's improving?
                # No, if we return inf, it might stop or behave badly.
                # Better to just let it run and check calls after?
                # But we must not exceed budget.
                # We can use a flag.
                pass
            
            # Actually, a cleaner way:
            # We don't need to wrap for counting if we just limit maxfev in minimize.
            # But we need to know the best value found.
            # minimize returns the best found.
            
            # Let's just use the budget limit in minimize.
            val = func(x)
            calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Quick random search to find a good starting point
        # Use 20% of budget for random search? No, Sphere is easy.
        # Let's try 5 random points.
        n_random = 5
        for _ in range(n_random):
            if calls >= budget:
                break
            x_rand = np.random.uniform(-5, 5, n)
            f_rand = func(x_rand)
            calls += 1
            if f_rand < best_f:
                best_f = f_rand
                best_x = x_rand.copy()

        # Phase 2: Nelder-Mead from best_x
        if calls < budget:
            # Set maxfev to remaining budget
            maxfev = budget - calls
            
            # Initial simplex: use a small step
            # If we start at best_x, a small step is good.
            # But if best_x is far from optimum, Nelder-Mead might get stuck in a flat region?
            # Sphere is convex, no flat regions.
            # Let's use a step size of 0.5 initially.
            x0 = best_x.copy()
            simplex = [x0]
            for i in range(n):
                xi = x0.copy()
                xi[i] += 0.5
                simplex.append(xi)
            
            res = minimize(
                wrapper, 
                x0, 
                method='Nelder-Mead', 
                options={
                    'maxfev': maxfev,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'initial_simplex': simplex,
                    'adaptive': True
                }
            )
            
            # Update best from result
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
            # Also check if the last point evaluated by wrapper was better
            # wrapper updates best_x, best_f directly.

        return best_x, best_f
