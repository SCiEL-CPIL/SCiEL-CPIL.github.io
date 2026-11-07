"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES style optimizer using scipy's differential_evolution or L-BFGS-B hybrid."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # We'll use a multi-start L-BFGS-B approach with a few random initializations
        # This is different from the (1+1)-ES approach.
        # We'll split the budget among multiple starts.
        
        n_starts = 5
        calls_per_start = max(1, (budget - n_starts) // n_starts)
        
        best_x = None
        best_f = np.inf
        
        for i in range(n_starts):
            # Random start within the box
            x0 = rng.uniform(-4.5, 4.5, dim)
            
            # Use L-BFGS-B with a limited number of function evaluations
            # Note: L-BFGS-B may use more calls than maxiter, so we need to be careful
            # We'll use a wrapper to count calls and stop if we exceed budget
            
            calls_used = 0
            total_calls = 0  # Track total calls across all starts
            
            def wrapped_func(x):
                nonlocal calls_used, total_calls
                if total_calls >= budget:
                    return 1e10  # Return a large value to discourage further exploration
                x_clipped = np.clip(x, -5, 5)
                val = func(x_clipped)
                calls_used += 1
                total_calls += 1
                return val
            
            try:
                # L-BFGS-B with bounds
                res = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': calls_per_start, 'maxfun': calls_per_start * 3}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass
            
            # If we've used all our budget, break
            if total_calls >= budget:
                break
        
        # If we have remaining budget, do a fine-tuning search around the best point
        if best_x is not None and total_calls < budget:
            remaining = budget - total_calls
            if remaining > 10:
                # Do a small CMA-ES or grid search around best_x
                # Let's do a simple random perturbation search
                sigma = 0.5
                for _ in range(remaining - 1):
                    if total_calls >= budget:
                        break
                    y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    total_calls += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y
                        sigma *= 1.2
                    else:
                        sigma *= 0.9
        
        if best_x is None:
            # Fallback: just return a random point
            best_x = rng.uniform(-5, 5, dim)
            best_f = func(best_x)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
