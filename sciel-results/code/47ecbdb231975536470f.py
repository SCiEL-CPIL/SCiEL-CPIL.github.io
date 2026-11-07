"""Rewritten optimizer for BBOB f6 (Attractive sector) using Multi-Start (1+1)-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB f6 (Attractive sector) in 5D with budget 500.
    
    Strategy:
      - Multi-Start (1+1)-ES.
      - 5 restarts, each with 100 function evaluations.
      - Each restart starts from a random point to explore different sectors.
      - Adaptive step size based on success rate (1/5 rule).
      - Time guard to ensure we finish within the limit.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        x_best = None
        f_best = float('inf')
        calls_used = 0

        # Multi-start strategy: 5 restarts of 100 calls each
        n_restarts = 5
        calls_per_restart = budget // n_restarts

        for r in range(n_restarts):
            # Check time
            if time.time() > deadline:
                break
            
            # Initialize current run
            # Start from a random point in the box
            x = rng.uniform(-5, 5, size=dim)
            fx = func(x)
            calls_used += 1
            
            # Update global best
            if fx < f_best:
                f_best = fx
                x_best = x.copy()

            # (1+1)-ES parameters for this restart
            sigma = 1.0
            n_success = 0
            n_total = 0
            window_size = 8

            # Run (1+1)-ES for this restart
            for i in range(calls_per_restart - 1):
                # Check time
                if time.time() > deadline:
                    return x_best, f_best
                
                # Generate offspring
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1

                if fy < fx:
                    x, fx = y, fy
                    n_success += 1
                    # Update global best
                    if fx < f_best:
                        f_best = fx
                        x_best = x.copy()
                n_total += 1

                # Adapt sigma based on success rate over the window
                if n_total >= window_size:
                    success_rate = n_success / n_total
                    # Target success rate around 1/5 = 0.2
                    if success_rate > 0.25:
                        sigma *= 1.15
                    elif success_rate < 0.15:
                        sigma *= 0.85
                    n_success = 0
                    n_total = 0

                # Keep sigma in reasonable bounds
                sigma = np.clip(sigma, 0.005, 5.0)

        return x_best, f_best
# EVOLVE-BLOCK-END
