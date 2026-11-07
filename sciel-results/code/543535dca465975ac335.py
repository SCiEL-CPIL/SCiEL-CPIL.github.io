"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-restart L-BFGS-B optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin within 300s limit

        # Count function calls
        call_count = [0]
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal best_x, best_f
            if call_count[0] >= self.budget:
                return best_f
            x_clipped = np.clip(x, -5, 5)
            fval = func(x_clipped)
            call_count[0] += 1
            if fval < best_f:
                best_f = fval
                best_x = x_clipped.copy()
            return fval

        # Generate initial points: combination of grid and random
        rng = np.random.default_rng(42)
        
        # Create a set of initial points
        init_points = []
        
        # Add center
        init_points.append(np.zeros(self.dim))
        
        # Add corners of a smaller box
        for i in range(2 ** self.dim):
            point = np.array([(-1 if (i >> j) & 1 else 1) * 3.0 for j in range(self.dim)])
            init_points.append(point)
        
        # Add random points
        while len(init_points) < 15:
            init_points.append(rng.uniform(-5, 5, self.dim))
        
        # Shuffle to mix
        rng.shuffle(init_points)
        
        # Run L-BFGS-B from each initial point
        bounds = [(-5, 5)] * self.dim
        
        for i, x0 in enumerate(init_points):
            if time.time() > deadline - 10:
                break
            if call_count[0] >= self.budget:
                break
            
            try:
                result = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'maxls': 20, 'ftol': 1e-12, 'gtol': 1e-8}
                )
            except Exception:
                continue
            
            if result.fun < best_f:
                best_f = result.fun
                best_x = np.clip(result.x, -5, 5)
        
        # Use remaining budget for a final (1+1)-ES polish if time allows
        if call_count[0] < self.budget and best_x is not None:
            remaining = self.budget - call_count[0]
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            
            for _ in range(remaining):
                if time.time() > deadline - 2:
                    break
                if call_count[0] >= self.budget:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = wrapped_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.9
        
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
