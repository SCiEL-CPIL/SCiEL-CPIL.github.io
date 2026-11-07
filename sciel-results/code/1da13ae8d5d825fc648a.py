"""Portfolio search optimizer for BBOB f20 (Schwefel) with explicit deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Multi-start (1+1)-ES with adaptive step size and time-based deadline guard.
    Designed for BBOB f20 (Schwefel) on [-5,5]^5 with budget 500.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin before 330s kill
        calls_made = 0
        best_x = None
        best_f = float('inf')

        def try_func(x):
            nonlocal calls_made
            calls_made += 1
            return func(x)

        # Multi-start strategy: try several different initializations
        # with different random seeds to escape local optima of Schwefel
        num_restarts = 5
        calls_per_restart = max(1, (self.budget - 1) // num_restarts)

        for restart in range(num_restarts):
            # Check deadline
            if time.time() > deadline - 5:
                break
            if calls_made >= self.budget:
                break

            # Different initialization strategy per restart
            seed = 42 + restart * 100
            rng = np.random.default_rng(seed)

            # For Schwefel, the global minimum is known to be around
            # x* = 4.209687... in each dimension for the standard version
            # But here it's shifted, so we use diverse starting points
            if restart == 0:
                x = rng.uniform(-5, 5, self.dim)
            elif restart == 1:
                # Start near the known Schwefel minimum region
                x = np.array([4.209687] * self.dim) + rng.normal(0, 0.5, self.dim)
                x = np.clip(x, -5, 5)
            elif restart == 2:
                # Start at origin
                x = np.zeros(self.dim) + rng.normal(0, 1.0, self.dim)
                x = np.clip(x, -5, 5)
            elif restart == 3:
                # Start at negative region
                x = np.array([-4.209687] * self.dim) + rng.normal(0, 0.5, self.dim)
                x = np.clip(x, -5, 5)
            else:
                # Random start
                x = rng.uniform(-5, 5, self.dim)

            fx = try_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 2.0
            min_sigma = 1e-6
            max_sigma = 10.0

            # Run (1+1)-ES for this restart
            for step in range(calls_per_restart):
                # Check deadline
                if time.time() > deadline - 5:
                    break
                if calls_made >= self.budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = try_func(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.3, max_sigma)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma = max(sigma * 0.8, min_sigma)

            # If we've made good progress, reduce remaining restarts
            if best_f < -100:  # Good enough, stop early
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
