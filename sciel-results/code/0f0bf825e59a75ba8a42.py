"""Optimizer for BBOB f24 (Lunacek bi-Rastrigin).
Strategy: Multi-start (1+1)-ES with momentum for global exploration and
local refinement. (1+1)-ES with step-size adaptation is robust to the
multimodality of the Lunacek bi-Rastrigin function.
"""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # 15s safety margin for 300s limit
        d = self.dim
        budget = self.budget

        # Best solution tracking
        best_x = None
        best_f = np.inf
        calls_made = 0

        def eval_func(x):
            nonlocal best_x, best_f, calls_made
            # Clip to box
            x_clip = np.clip(x, -5.0, 5.0)
            fx = func(x_clip)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clip.copy()
            return fx

        # Multi-start (1+1)-ES with momentum
        # Generate diverse starting points using Sobol sequence
        n_starts = min(20, max(5, budget // 10))
        rng = np.random.default_rng(42)
        
        # Sobol for diverse starts
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=d, scramble=True, seed=42)
            sobol_samples = sampler.random(n_starts)
            start_points = sobol_samples * 10.0 - 5.0
        except Exception:
            start_points = rng.uniform(-5, 5, size=(n_starts, d))

        # Ensure we try the origin
        start_points[0] = np.zeros(d)

        # Run (1+1)-ES from each starting point
        for start_idx in range(n_starts):
            if calls_made >= budget - 5:
                break
            if time.time() > deadline:
                break

            x = start_points[start_idx].copy()
            fx = eval_func(x)
            
            # (1+1)-ES parameters
            sigma = 2.0  # initial step size
            momentum = np.zeros(d)
            success_count = 0
            total_steps = 0
            max_steps = max(10, (budget - calls_made) // max(1, n_starts - start_idx))
            
            for step in range(max_steps):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                # Generate candidate
                direction = rng.normal(0, 1, size=d)
                # Add momentum
                momentum = 0.5 * momentum + 0.5 * direction
                x_candidate = x + sigma * momentum
                
                fx_candidate = eval_func(x_candidate)
                
                # Update
                if fx_candidate < fx:
                    # Success
                    x = x_candidate
                    fx = fx_candidate
                    success_count += 1
                    # Increase step size slightly
                    sigma *= 1.05
                else:
                    # Failure
                    # Decrease step size
                    sigma *= 0.95
                    # Reset momentum partially
                    momentum *= 0.5
                
                total_steps += 1
                if total_steps > 50:
                    # Reset momentum to prevent drift
                    momentum = np.zeros(d)
                    total_steps = 0
        
        return best_x, best_f
