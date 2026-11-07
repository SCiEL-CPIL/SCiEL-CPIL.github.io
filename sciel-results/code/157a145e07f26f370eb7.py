"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with time guard for Rosenbrock-style optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * dim
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        # Counting wrapper to track function evaluations
        count = [0]
        def counting_func(x):
            count[0] += 1
            return func(np.clip(x, -5.0, 5.0))

        # Generate diverse starting points
        rng = np.random.default_rng(42)
        
        # Strategy: mix of structured and random starts
        starts = []
        
        # Add some structured points
        starts.append(np.zeros(dim))
        starts.append(np.ones(dim) * 3.0)
        starts.append(np.ones(dim) * -3.0)
        starts.append(np.array([1.0, 1.0, 1.0, 1.0, 1.0]) * 2.0)
        starts.append(np.array([1.0, 1.0, 1.0, 1.0, 1.0]) * -2.0)
        
        # Add random points
        n_random = 20
        for _ in range(n_random):
            starts.append(rng.uniform(-5.0, 5.0, dim))
        
        # Sort starts by a heuristic: prefer points closer to typical Rosenbrock minima region
        # Rosenbrock minimum is at (1,1,1,1,1) but shifted, so use norm as rough guide
        # Actually, just use them as-is for diversity
        
        best_x = None
        best_f = np.inf
        
        for x0 in starts:
            if count[0] >= budget:
                break
            if time.time() > deadline:
                break
            
            # Run L-BFGS-B from this start
            try:
                result = minimize(
                    counting_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if result.success or result.fun < best_f:
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                        
            except Exception:
                continue
            
            # Check budget
            if count[0] >= budget:
                break
            
            # Check time
            if time.time() > deadline:
                break
        
        # If we still have budget, do a few more random restarts
        if best_x is not None and count[0] < budget:
            while count[0] < budget and time.time() < deadline:
                x0 = rng.uniform(-5.0, 5.0, dim)
                try:
                    result = minimize(
                        counting_func,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-7}
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    continue
        
        # If no valid result found, return a default
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
