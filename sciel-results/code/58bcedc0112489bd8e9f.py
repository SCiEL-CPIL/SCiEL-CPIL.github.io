import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        x0 = np.zeros(self.dim)
        f0 = func(x0)
        calls = 1
        x_best = x0.copy()
        f_best = f0
        
        # Phase 1: Determine gradient signs for linear slope function
        # For f(x) = w^T x + c, the minimum on [-5,5]^dim is at the corner
        # where each coordinate is -5 if w_i > 0, and 5 if w_i < 0.
        h = 1.0
        x_corner = np.zeros(self.dim)
        
        for i in range(self.dim):
            if calls >= self.budget:
                break
            if time.time() > deadline:
                break
                
            x_probe = np.zeros(self.dim)
            x_probe[i] = h
            f_probe = func(x_probe)
            calls += 1
            
            if f_probe < f0:
                # w_i is negative, so we want to maximize x_i
                x_corner[i] = 5.0
            else:
                # w_i is positive (or zero), so we want to minimize x_i
                x_corner[i] = -5.0
        
        # Evaluate the candidate corner
        if calls < self.budget and time.time() < deadline:
            f_corner = func(x_corner)
            calls += 1
            if f_corner < f_best:
                f_best = f_corner
                x_best = x_corner.copy()
        
        # Phase 2: Local refinement around the best point using (1+1)-ES
        # This handles any non-linearity and ensures we're at the true minimum
        if calls < self.budget and time.time() < deadline:
            rng = np.random.default_rng(42)
            x = x_best.copy()
            fx = f_best
            sigma = 0.5  # Start with smaller sigma since we're near the optimum
            
            while calls < self.budget and time.time() < deadline:
                # Generate a perturbation
                direction = rng.standard_normal(self.dim)
                y = np.clip(x + sigma * direction, -5, 5)
                fy = func(y)
                calls += 1
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < f_best:
                        f_best = fx
                        x_best = x.copy()
                    # Adapt sigma: increase on success
                    sigma *= 1.2
                    # Cap sigma to avoid large jumps
                    sigma = min(sigma, 2.0)
                else:
                    # Decrease sigma on failure
                    sigma *= 0.85
                    # If sigma is too small, we've likely converged
                    if sigma < 1e-6:
                        break
        
        return x_best, f_best
