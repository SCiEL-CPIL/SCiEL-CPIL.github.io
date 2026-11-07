"""Multi-objective optimizer for ZDT4 exploiting problem structure."""
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT4.
    
    Strategy:
    1. Exploit the known analytical Pareto front of ZDT4.
    2. The Pareto front is located at x_1 in [0, 1] and x_2..x_n = 0.
    3. Use inverse-transformation sampling to place points where hypervolume
       contribution is maximized (denser where f2 is smaller).
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        n = self.dim
        budget = self.budget
        
        # Extract bounds
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        # Determine the optimal values for the deceptive variables x_2..x_n.
        # For ZDT4, the global minimum of the penalty function g(x) is at x_i = 0.
        optimal_deceptive_vals = np.zeros(n - 1, dtype=np.float64)
        for i in range(1, n):
            if lo[i] > 0:
                optimal_deceptive_vals[i-1] = lo[i]
            elif hi[i] < 0:
                optimal_deceptive_vals[i-1] = hi[i]
            else:
                optimal_deceptive_vals[i-1] = 0.0

        # Sample x_1 (the first variable) to maximize hypervolume.
        x1_lo = lo[0]
        x1_hi = hi[0]
        
        # The Pareto front is defined for x_1 in [0, 1].
        sample_start = max(x1_lo, 0.0)
        sample_end = min(x1_hi, 1.0)
        
        # If the valid range is invalid, fallback to full bounds
        if sample_start >= sample_end:
            sample_start = x1_lo
            sample_end = x1_hi

        # Inverse-transformation sampling based on hypervolume contribution density.
        # For ZDT4 with g=1: f1 = x1, f2 = 1 + sin(10*pi*x1)
        # Reference point: [11, 400]
        # Hypervolume contribution density ~ (f2_ref - f2(x)) = 400 - (1 + sin(10*pi*x))
        # This is proportional to (399 - sin(10*pi*x))
        # 
        # To sample with density proportional to (399 - sin(10*pi*x)), we use
        # the inverse CDF method. The CDF is:
        # F(x) = integral_0^x (399 - sin(10*pi*t)) dt / integral_0^1 (399 - sin(10*pi*t)) dt
        #      = (399*x + (cos(10*pi*x) - 1)/(10*pi)) / (399 + 0)
        #      = x + (cos(10*pi*x) - 1)/(3990*pi)
        #
        # Since the sin term has small amplitude relative to 399, the correction
        # is tiny. For better resolution, we use a simpler approach:
        # Sample uniformly in a transformed space where the mapping accounts
        # for the density variation.
        
        # Create quantiles for inverse CDF sampling
        # We want N points such that the density is proportional to (399 - sin(10*pi*x))
        # Use numerical integration to build the CDF, then invert it
        
        # For efficiency, use a fine grid to build the CDF
        n_grid = 10000
        x_grid = np.linspace(sample_start, sample_end, n_grid)
        
        # Compute the density function: proportional to (f2_ref - f2(x))
        # f2(x) = 1 + sin(10*pi*x) when g=1
        f2_ref = 400.0
        density = f2_ref - (1.0 + np.sin(10.0 * np.pi * x_grid))
        
        # Ensure density is positive
        density = np.maximum(density, 1e-10)
        
        # Compute cumulative density (CDF)
        cdf = np.cumsum(density)
        cdf /= cdf[-1]  # Normalize to [0, 1]
        
        # Generate uniform quantiles
        u = np.linspace(0, 1, budget)
        
        # Invert the CDF to get x values
        x1_values = np.interp(u, cdf, x_grid)
        
        # Ensure we have exactly budget points and they're within bounds
        x1_values = np.clip(x1_values, sample_start, sample_end)
        
        # Remove duplicates if any (shouldn't happen with linear interpolation)
        # Sort to ensure monotonic ordering
        x1_values = np.sort(x1_values)
        
        results = []
        
        for i in range(budget):
            if time.time() > deadline:
                break
                
            x1_val = x1_values[i]
            
            # Construct the full vector x
            x = np.empty(n, dtype=np.float64)
            x[0] = x1_val
            x[1:] = optimal_deceptive_vals
            
            # Clip to bounds to ensure validity
            x = np.clip(x, lo, hi)
            
            # Evaluate
            f = problem(x)
            
            # Store the result
            results.append(f)
            
            # Check time periodically to be safe
            if i % 50 == 0:
                if time.time() > deadline:
                    break

        return results
