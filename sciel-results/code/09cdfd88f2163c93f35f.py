import numpy as np
import time
import itertools

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        budget = self.budget
        dim = self.dim

        calls_made = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        def eval_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return float('inf')
            # Check time limit strictly before calling
            if time.time() > deadline:
                return float('inf')
            
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = np.array(x)
            return fx

        # Phase 1: Halton Sequence Sampling
        # Use a Halton sequence to systematically cover the domain.
        # This is a low-discrepancy sequence that provides better space-filling properties
        # than a standard grid or random sampling, especially in higher dimensions.
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i = i // base
            return r

        # Sample points using Halton sequence
        # We want to cover the domain [-5, 5]^dim.
        # The Halton sequence generates points in [0, 1]^dim.
        # We map them to [-5, 5]^dim.
        
        # Determine number of samples.
        # We have 500 budget. Let's use 200 for global exploration.
        # This leaves 300 for local refinement.
        num_samples = min(200, budget - 50)
        
        halton_results = []
        
        for i in range(1, num_samples + 1):
            if calls_made >= budget - 50: # Reserve calls for optimization
                break
            if time.time() > deadline:
                break
            
            x = np.array([halton(i, base) for base in range(2, dim + 2)])
            # Map from [0, 1] to [-5, 5]
            x = x * 10.0 - 5.0
            
            fx = eval_func(x)
            halton_results.append((fx, x))

        if not halton_results:
            # Fallback if no points were evaluated
            x = np.zeros(dim)
            eval_func(x)
            return best_x, best_f

        # Sort to find the best point
        halton_results.sort(key=lambda t: t[0])
        best_halton_f, best_halton_x = halton_results[0]

        # Phase 2: Local Optimization using (1+1)-ES
        # This is a robust derivative-free method that adapts step size.
        
        if time.time() < deadline and calls_made < budget - 10:
            rng = np.random.default_rng(42)
            x_curr = best_halton_x.copy()
            f_curr = best_halton_f
            sigma = 1.0
            
            # Run (1+1)-ES for remaining budget
            while calls_made < budget - 5 and time.time() < deadline:
                y = x_curr + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = eval_func(y)
                
                if fy < f_curr:
                    x_curr = y
                    f_curr = fy
                    sigma *= 1.2
                else:
                    sigma *= 0.8

        return best_x, best_f
