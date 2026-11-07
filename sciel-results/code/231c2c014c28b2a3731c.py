"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with diverse random initializations."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        # Track best solution found
        best_x = None
        best_f = np.inf
        call_count = 0

        # Wrapper to count function calls and enforce deadline
        def wrapped_func(x):
            nonlocal call_count
            call_count += 1
            if call_count > budget:
                raise ValueError("Budget exceeded")
            x_clipped = np.clip(x, -5.0, 5.0)
            return func(x_clipped)

        # Generate diverse random starting points
        rng = np.random.default_rng(42)
        
        # Strategy: use a mix of random points and structured points
        # For Rosenbrock, the valley is narrow, so we want points along the valley
        # but also diverse enough to cover different basins
        
        n_starts = 20  # Use 20 diverse starts (4% of budget for initial evals)
        starts = []
        
        # Some purely random
        for i in range(n_starts // 2):
            starts.append(rng.uniform(-5, 5, dim))
        
        # Some structured: vary one coordinate at a time from center
        for i in range(n_starts // 2):
            x = np.zeros(dim)
            # Place a large value in one random dimension
            d = i % dim
            x[d] = rng.uniform(-5, 5)
            starts.append(x)
        
        # Sort starts by distance from origin (Rosenbrock min is near origin in rotated space)
        # Actually, let's just use them as is since the rotation is hidden
        
        # For each start, run L-BFGS-B
        bounds = [(-5.0, 5.0)] * dim
        
        for x0 in starts:
            if time.time() > deadline:
                break
            if call_count >= budget:
                break
                
            try:
                # Set max iterations to use remaining budget
                remaining = budget - call_count
                max_iter = min(remaining, 100)  # Cap per-start iterations
                
                result = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': max_iter, 'maxls': 20}
                )
                
                if result.success or result.fun < best_f:
                    best_x = np.clip(result.x, -5.0, 5.0)
                    best_f = result.fun
            except Exception:
                # If L-BFGS-B fails, try a simple gradient-free approach
                pass

        # If we still have budget, do a final polish with more iterations
        if best_x is not None and call_count < budget and time.time() < deadline:
            remaining = budget - call_count
            try:
                result = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': remaining, 'maxls': 50, 'ftol': 1e-15, 'gtol': 1e-15}
                )
                if result.fun < best_f:
                    best_x = np.clip(result.x, -5.0, 5.0)
                    best_f = result.fun
            except Exception:
                pass

        # If no valid solution found, return a default
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
