"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with derivative-free fallback for Discus (f11)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Count evaluations
        calls = [0]
        
        def wrapped_func(x):
            calls[0] += 1
            x = np.clip(x, -5.0, 5.0)
            return float(func(x))
        
        # Generate a coarse grid of starting points in [-5, 5]^dim
        # Use a mix of deterministic grid and random points
        n_starts = min(10, budget // 50)  # Reserve budget for each start
        if n_starts < 1:
            n_starts = 1
        
        # Deterministic grid points
        grid_points = []
        # Center point
        grid_points.append(np.zeros(dim))
        # Points at ±2 in each dimension
        for i in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[i] = sign * 2.0
                grid_points.append(p)
        
        # Add some random points to fill up to n_starts
        while len(grid_points) < n_starts:
            grid_points.append(rng.uniform(-4, 4, dim))
        
        # If we have more than n_starts, randomly select
        if len(grid_points) > n_starts:
            indices = rng.choice(len(grid_points), n_starts, replace=False)
            grid_points = [grid_points[i] for i in indices]
        
        best_x = None
        best_f = np.inf
        
        # Try L-BFGS-B from each starting point
        evals_per_run = max(10, budget // (n_starts + 2))
        
        for x0 in grid_points:
            if calls[0] >= budget - 5:
                break
            
            x0 = np.clip(x0, -5.0, 5.0)
            f0 = wrapped_func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            
            # Run L-BFGS-B from this point
            try:
                result = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'maxiter': 100,
                        'maxfun': evals_per_run - 1,  # Subtract the initial eval
                        'gtol': 1e-8,
                        'ftol': 1e-12
                    }
                )
                if result.success or result.fun < best_f:
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -5.0, 5.0)
            except Exception:
                pass
            
            # Early stop if we've used most of the budget
            if calls[0] >= budget - 10:
                break
        
        # If we still have budget left, do a derivative-free refinement
        remaining = budget - calls[0]
        if remaining > 5 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            
            # Adaptive step size (1+1)-ES refinement
            for _ in range(remaining - 1):
                if calls[0] >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = wrapped_func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.85
                
                # Stop if we've converged
                if sigma < 1e-6:
                    break
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Ensure we made at least one call
        if calls[0] == 0:
            x0 = np.zeros(dim)
            best_f = wrapped_func(x0)
            best_x = x0.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
