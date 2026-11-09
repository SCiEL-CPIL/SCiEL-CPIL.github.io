# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Adaptive (1+1)-ES with per-dimension step sizes, stall detection, and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        n = self.dim
        budget = self.budget

        calls_made = [0]
        best_f = np.inf
        best_x = np.zeros(n)

        def wrapped_func(x):
            nonlocal best_f, best_x
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
            calls_made[0] += 1
            if calls_made[0] > budget:
                raise Exception("Budget exceeded")
            
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        rng = np.random.default_rng(42)
        
        # Per-dimension sigma adaptation
        sigma = np.full(n, 1.5)
        min_sigma = 1e-4
        max_sigma = 5.0
        
        # Start at center
        x = np.zeros(n)
        fx = wrapped_func(x)
        
        window_size = 5
        successes = np.zeros(n, dtype=int)
        steps_in_window = 0
        stall_threshold = 30
        steps_without_improvement = 0

        while calls_made[0] < budget:
            if time.time() > deadline:
                break

            # Generate candidate with per-dimension step sizes
            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = wrapped_func(y)
            
            if fy < fx:
                x, fx = y, fy
                steps_without_improvement = 0
                # Increase sigma for successful dimensions
                sigma = np.minimum(sigma * 1.15, max_sigma)
            else:
                steps_without_improvement += 1
                # Decrease sigma for failed dimensions
                sigma = np.maximum(sigma * 0.88, min_sigma)
            
            steps_in_window += 1
            
            # Adapt step size based on 1/5 rule per dimension
            if steps_in_window >= window_size:
                success_rate = successes / steps_in_window
                for i in range(n):
                    if success_rate[i] > 1/5:
                        sigma[i] = min(sigma[i] * 1.05, max_sigma)
                    else:
                        sigma[i] = max(sigma[i] * 0.95, min_sigma)
                successes = np.zeros(n, dtype=int)
                steps_in_window = 0
            
            # Track successes per dimension
            if fy < fx:
                successes += 1

            # Stall detection
            if steps_without_improvement > stall_threshold:
                # Restart with a random point, biased towards center
                x = rng.uniform(-3, 3, n)
                fx = wrapped_func(x)
                sigma = np.full(n, 1.5)
                steps_without_improvement = 0
                successes = np.zeros(n, dtype=int)
                steps_in_window = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
