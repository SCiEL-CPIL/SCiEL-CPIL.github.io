"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    """Multi-start Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        best_x = None
        best_f = float('inf')
        calls_used = 0
        max_calls = budget

        def wrapped_func(x):
            nonlocal calls_used, best_f, best_x
            if calls_used >= max_calls:
                return best_f
            x_clipped = np.clip(x, -5.0, 5.0)
            val = float(func(x_clipped))
            calls_used += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        # Strategy: Multi-start Nelder-Mead
        # 1. Generate diverse starting points
        # 2. Run local optimization from each, sharing the budget
        
        # Number of starts: balance between exploration (many starts) and exploitation (budget per start)
        # For dim=5, budget=500.
        # Nelder-Mead typically needs ~10-50 evaluations to converge locally.
        # Let's try 10-20 starts.
        
        # Use Latin Hypercube Sampling for initial points
        def lhs_sample(n_samples, dim, bounds):
            # bounds is list of (low, high)
            samples = np.zeros((n_samples, dim))
            for i in range(dim):
                # Shuffle indices to create stratified sampling
                idx = np.arange(n_samples)
                np.random.shuffle(idx)
                # Map to [0, 1]
                u = (idx + np.random.rand(n_samples)) / n_samples
                # Map to bounds
                low, high = bounds[i]
                samples[:, i] = low + u * (high - low)
            return samples

        bounds = [(-5.0, 5.0)] * dim
        
        # Determine number of starts
        # Reserve some budget for the best start to refine? 
        # Or just split evenly?
        # Let's start with 10 starts.
        n_starts = 10
        if budget < 100:
            n_starts = 5
        elif budget > 1000:
            n_starts = 20
            
        # Ensure we have enough budget per start to do something meaningful
        # Nelder-Mead minimum is ~dim+1 for simplex, but needs more to converge.
        # Let's say min 20 calls per start.
        if budget // n_starts < 20:
            n_starts = max(1, budget // 20)

        # Generate initial points
        initial_points = lhs_sample(n_starts, dim, bounds)
        
        # Shuffle the order of starts to randomize which one gets the leftover budget if we stop early
        np.random.shuffle(initial_points)

        # Run Nelder-Mead from each start
        for x0 in initial_points:
            if calls_used >= max_calls:
                break
            
            # Calculate remaining budget for this start
            remaining = max_calls - calls_used
            if remaining < 5:
                break
                
            # Limit maxfev for this specific run to not starve later starts too much?
            # Actually, if one start finds a good solution, we might want to spend more there.
            # But for a robust multi-start, we should give each start a fair share initially.
            # Let's give each start a budget of budget/n_starts, but allow the last one to use the rest?
            # Or just let it run until budget is exhausted?
            # If the first start is bad, it might waste budget.
            # Let's cap each start to budget // n_starts + some buffer.
            # Simpler: just run with maxfev=remaining, but break if no improvement after some steps?
            # Nelder-Mead doesn't easily report "no improvement" without custom callback.
            
            # Let's just run Nelder-Mead with the remaining budget, but we might want to limit it
            # to prevent one start from eating everything if it's slow.
            # However, Nelder-Mead is fast. 
            # Let's set maxfev to a reasonable chunk, e.g., 50, or remaining if small.
            # If we have 500 budget and 10 starts, 50 per start is good.
            maxfev_this = min(remaining, 50)
            
            try:
                res = minimize(
                    wrapped_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200, 
                        'maxfev': maxfev_this, 
                        'xatol': 1e-8, 
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
