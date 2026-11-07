# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.perf_counter()
        deadline = t_start + 280.0  # Safety margin

        rng = np.random.default_rng(42)
        
        # Track best solution
        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return float('inf')
            if time.perf_counter() > deadline:
                return float('inf')
            calls_made += 1
            try:
                f_val = float(func(x))
            except Exception:
                return float('inf')
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Radial Sweep Initialization
        # Sample points along axes and random directions to find the needle
        directions = []
        for i in range(dim):
            for sign in [1, -1]:
                d = np.zeros(dim)
                d[i] = sign
                directions.append(d)
        
        for _ in range(10):
            d = rng.standard_normal(dim)
            norm = np.linalg.norm(d)
            if norm > 1e-10:
                d /= norm
                directions.append(d)
                
        radii = [0.5, 1.0, 2.0, 3.0, 4.0]
        
        for d in directions:
            for r in radii:
                if calls_made >= budget or time.perf_counter() > deadline:
                    break
                x = np.clip(r * d, -5, 5)
                f = evaluate(x)
                if f == float('inf'):
                    break

        if calls_made >= budget or time.perf_counter() > deadline:
            return best_x, best_f

        # Phase 2: Use Nelder-Mead from scipy, which adapts well to anisotropic curvature
        # We'll run multiple Nelder-Mead instances from different starting points
        # to handle the multimodal nature and find the global minimum
        
        # Collect candidate starting points
        start_points = [best_x.copy()]
        
        # Add some random points around the best found so far
        for _ in range(3):
            perturbation = rng.standard_normal(dim) * 0.5
            x_start = np.clip(best_x + perturbation, -5, 5)
            start_points.append(x_start)
        
        # Add a few more diverse starting points
        for _ in range(2):
            x_start = rng.uniform(-4, 4, size=dim)
            start_points.append(x_start)
        
        # Run Nelder-Mead from each starting point
        for x0 in start_points:
            if calls_made >= budget or time.perf_counter() > deadline:
                break
            
            try:
                # Use Nelder-Mead with adaptive settings
                result = minimize(
                    lambda x: evaluate(x),
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                )
            except Exception:
                continue
            
            if calls_made >= budget or time.perf_counter() > deadline:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
