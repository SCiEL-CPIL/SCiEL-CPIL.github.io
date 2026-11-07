"""
Optimizer for BBOB f24 (Lunacek bi-Rastrigin) using a hybrid approach:
1. Dense Latin Hypercube Sampling (LHS) for global basin identification.
2. Powell's method for local exploitation (exploits separable structure).
3. Adaptive restarts if stagnation is detected.
"""
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        # Helper to update best
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Latin Hypercube Sampling (LHS)
        # LHS ensures coverage of the entire space, which is crucial for 
        # identifying the correct global basin in bi-modal functions.
        # We use a stratified sampling approach.
        
        # Determine number of samples for LHS
        # We want to use a significant portion of the budget for global exploration
        # but leave enough for local refinement.
        # Let's use 50% of the budget for LHS if possible, but at least 50 samples.
        n_lhs = max(50, budget // 2)
        
        # Generate LHS samples
        # For each dimension, divide the range into n_lhs intervals and sample one point per interval
        # Then shuffle the indices for each dimension to decorrelate
        lhs_samples = np.zeros((n_lhs, dim))
        for d in range(dim):
            # Create intervals
            intervals = np.linspace(-5, 5, n_lhs + 1)
            # Sample one point from each interval
            # Random offset within interval
            offsets = np.random.uniform(0, 1, size=n_lhs)
            points = intervals[:-1] + offsets * (intervals[1] - intervals[0])
            # Shuffle to break correlation
            np.random.shuffle(points)
            lhs_samples[:, d] = points

        # Evaluate LHS samples
        for i in range(n_lhs):
            if calls >= budget or time.time() > deadline:
                break
            x = lhs_samples[i]
            f = func(x)
            calls += 1
            update_best(x, f)

        if best_x is None:
            # Fallback if no calls were made (should not happen)
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls += 1

        # Phase 2: Local Refinement with Powell's Method
        # Powell's method is particularly effective for functions with separable structure
        # like Lunacek's bi-Rastrigin, as it performs line searches along directions
        # that adapt to the function's geometry without requiring gradient estimates.
        
        # Define the objective function for Powell's method
        def objective(x):
            nonlocal calls
            if calls >= budget or time.time() > deadline:
                return best_f
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # Run Powell's method with tight tolerances for high-precision convergence
        options = {
            'maxiter': 500,
            'xtol': 1e-10,
            'ftol': 1e-10,
            'disp': False
        }

        try:
            result = minimize(objective, best_x, method='Powell', options=options)
            if result.fun < best_f:
                best_f = result.fun
                best_x = np.clip(result.x, -5, 5).copy()
        except Exception:
            pass

        # If we still have budget, try additional Powell restarts from perturbed best_x
        # to escape potential local minima or refine convergence from different starting points
        
        if calls < budget and time.time() < deadline:
            # Try several restarts with different perturbations
            for restart in range(5):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Perturb best_x with decreasing magnitude
                perturb_scale = 0.2 * (0.5 ** restart)
                x_start = best_x + np.random.normal(0, perturb_scale, size=dim)
                x_start = np.clip(x_start, -5, 5)
                
                try:
                    result = minimize(objective, x_start, method='Powell', options=options)
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -5, 5).copy()
                except Exception:
                    pass

        return best_x, best_f
