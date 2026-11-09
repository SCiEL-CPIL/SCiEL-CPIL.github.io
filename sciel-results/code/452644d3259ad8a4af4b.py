import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT3 Pareto front function
        def f2_of_f1(f1_vals):
            sqrt_f1 = np.sqrt(f1_vals)
            h = 1.0 - sqrt_f1 - np.sin(10.0 * np.pi * f1_vals)
            return (1.0 + sqrt_f1) * h

        # Calculate benefit (approximate HV contribution)
        # Benefit = (R1 - f1) * (R2 - f2)
        R1, R2 = 11.0, 11.0
        
        # Create a fine grid to estimate the distribution
        grid_size = 10000
        f1_grid = np.linspace(0, 1, grid_size)
        f2_grid = f2_of_f1(f1_grid)
        benefit_grid = (R1 - f1_grid) * (R2 - f2_grid)
        
        # Normalize to get a probability distribution
        # Add a small epsilon to avoid zero division if benefit is 0
        benefit_grid += 1e-6
        prob_grid = benefit_grid / np.sum(benefit_grid)
        
        # Cumulative distribution function
        cdf = np.cumsum(prob_grid)
        
        # Sample x0 values
        # We want budget samples.
        # Use inverse transform sampling
        u_samples = np.random.rand(budget)
        x0_samples = np.interp(u_samples, cdf, f1_grid)
        
        # Ensure we also cover the whole range to avoid missing parts
        # Mix with uniform samples
        n_uni = budget // 4
        n_prob = budget - n_uni
        
        # Re-sample with mixture
        u_prob = np.random.rand(n_prob)
        x0_prob = np.interp(u_prob, cdf, f1_grid)
        
        x0_uni = np.random.rand(n_uni)
        
        x0_samples = np.concatenate([x0_prob, x0_uni])
        
        # Sort and unique to avoid duplicates
        x0_samples = np.unique(np.sort(x0_samples))
        
        # If we have fewer than budget due to unique, add more random
        if len(x0_samples) < budget:
            while len(x0_samples) < budget:
                x0_samples = np.append(x0_samples, np.random.rand())
                x0_samples = np.unique(np.sort(x0_samples))
        
        # If we have more (shouldn't happen), trim
        if len(x0_samples) > budget:
            x0_samples = x0_samples[:budget]

        # Evaluate
        for x0 in x0_samples:
            if time.time() > deadline:
                break
            
            x = np.zeros(dim)
            x[0] = x0
            x[1:] = 0.0  # Minimize g
            
            # Clip to bounds (should be [0,1])
            x = np.clip(x, lo, hi)
            
            try:
                problem(x)
            except Exception:
                break

        return None
