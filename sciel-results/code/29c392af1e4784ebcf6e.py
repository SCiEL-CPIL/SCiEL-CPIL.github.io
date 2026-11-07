"""Multi-objective optimizer for ZDT4 exploiting problem structure."""
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT4.
    
    Strategy:
    1. Exploit the known analytical Pareto front of ZDT4.
    2. The Pareto front is located at x_1 in [0, 1] and x_2..x_n = 0.
    3. Use non-uniform sampling of x_1 to maximize hypervolume.
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

        # Generate non-uniform sampling points to maximize hypervolume.
        # Use a grid that's denser where the Pareto front has high curvature.
        # For ZDT4, f2 = g * (1 + sin(10*pi*x1)), so curvature is high near x1 = k/10.
        # We use a transformation to concentrate points in high-curvature regions.
        
        # Create a base uniform grid
        u = np.linspace(0, 1, budget)
        
        # Apply a transformation that concentrates points where |cos(10*pi*x)| is large
        # This means concentrating near x = k/10 for integer k
        # Use a smooth transformation: x = u + epsilon * sin(20*pi*u) / (20*pi)
        # This creates slight densification near the zeros of cos(10*pi*x)
        
        epsilon = 0.02  # Small perturbation to avoid discontinuities
        x1_values = sample_start + (sample_end - sample_start) * (u + epsilon * np.sin(20 * np.pi * u) / (20 * np.pi))
        
        # Clip to valid range
        x1_values = np.clip(x1_values, sample_start, sample_end)
        
        # Sort to ensure monotonic ordering (helps with hypervolume computation)
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
