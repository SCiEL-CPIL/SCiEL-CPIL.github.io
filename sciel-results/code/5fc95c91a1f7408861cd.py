import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0 # 10s safety margin
        
        # Fixed seed for reproducibility across instances? 
        # The prompt says "hidden seeded shift... instances apply". 
        # Usually, in these tasks, we should use a fixed seed to ensure 
        # the search pattern is consistent, but the function itself changes.
        # Using a fixed seed for the sampler is standard.
        rng = np.random.default_rng(42)
        
        # 1. Global Exploration: Latin Hypercube Sampling
        # We want to use a significant portion of the budget for exploration 
        # to find the correct basin, then exploit.
        # Let's allocate ~50 points for LHS and the rest for Nelder-Mead.
        # 50 points in 5D is a reasonable spread.
        n_lhs = 50
        lhs_points = self._generate_lhs(n_lhs, self.dim, rng)
        
        best_x = None
        best_f = np.inf
        all_points = []
        
        # Evaluate LHS points
        for i in range(n_lhs):
            if time.time() > deadline:
                break
            x = lhs_points[i]
            fx = func(x)
            all_points.append((x, fx))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        calls_made = len(all_points)
        
        if best_x is None:
            # Fallback if something went wrong
            x = np.zeros(self.dim)
            return x, func(x)

        # 2. Local Exploitation: Nelder-Mead
        # Use the remaining budget for local search.
        # Nelder-Mead options: maxiter, maxfev.
        # We have budget - calls_made remaining.
        remaining_budget = self.budget - calls_made
        
        if remaining_budget > 10:
            # Define the objective for scipy
            def objective(x):
                if time.time() > deadline:
                    return best_f # Return best found so far to stop optimization
                return func(x)
                
            # Initial simplex
            # Nelder-Mead generates its own simplex, but we can provide an initial point.
            # The default simplex size is 0.05 * x + 0.00025.
            # Since we are in [-5, 5], a slightly larger initial step might help.
            
            try:
                res = minimize(
                    objective, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': 200, # Safety cap on iterations
                        'maxfev': remaining_budget - 5, # Leave a few calls for safety
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
                    
            except Exception:
                pass # Keep the best from LHS if NM fails

        return best_x, best_f

    def _generate_lhs(self, n_samples, dim, rng):
        """
        Generate Latin Hypercube Samples.
        """
        # For each dimension, divide the range [-5, 5] into n_samples bins.
        # Pick one point from each bin randomly.
        points = np.zeros((n_samples, dim))
        
        # Create random permutations for each dimension
        for d in range(dim):
            # Indices for the bins
            idx = rng.permutation(n_samples)
            # Random offset within the bin
            offsets = rng.random(n_samples)
            
            # Bin width
            bin_width = 10.0 / n_samples
            
            # Calculate coordinates
            # (idx * bin_width) + (offsets * bin_width) - 5
            points[:, d] = (idx * bin_width) + (offsets * bin_width) - 5.0
            
        return points
