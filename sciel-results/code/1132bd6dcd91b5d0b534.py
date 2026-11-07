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
        deadline = start_time + 280.0  # 300s limit, 20s safety margin
        rng = np.random.default_rng(0)

        best_x = None
        best_f = np.inf

        def try_func(x):
            nonlocal best_x, best_f
            try:
                fx = func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx
            except Exception:
                return np.inf

        # Multi-start strategy with adaptive (1+1)-ES
        # Use several restarts to handle the rugged landscape of Schwefel
        
        # First, do a coarse random search to get an initial good point
        # Then refine with (1+1)-ES from multiple starting points
        
        num_restarts = 5
        calls_per_restart = (self.budget - 1) // num_restarts
        remaining_calls = self.budget - 1

        for restart in range(num_restarts):
            if remaining_calls <= 0:
                break
            if time.time() > deadline - 10:  # Check deadline with margin
                break

            # Initialize starting point
            if restart == 0:
                # Start from center or a random point
                x = rng.uniform(-5, 5, self.dim)
            else:
                # For subsequent restarts, perturb the best found so far
                if best_x is not None:
                    x = np.clip(best_x + 2.0 * rng.standard_normal(self.dim), -5, 5)
                else:
                    x = rng.uniform(-5, 5, self.dim)

            fx = try_func(x)
            remaining_calls -= 1
            if remaining_calls <= 0:
                break

            sigma = 1.5  # Initial step size
            # Adaptive (1+1)-ES with 1/5 success rule
            successes = 0
            trials = 0

            while remaining_calls > 0:
                if time.time() > deadline - 5:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = try_func(y)
                remaining_calls -= 1
                trials += 1

                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    sigma *= 1.2
                else:
                    sigma *= 0.85

                # Check if we've made enough trials to adapt sigma
                if trials >= 10:
                    success_rate = successes / trials
                    if success_rate > 0.5:
                        sigma *= 1.5
                    elif success_rate < 0.1:
                        sigma *= 0.5
                    successes = 0
                    trials = 0

                # Early stopping if sigma is very small
                if sigma < 1e-6:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
