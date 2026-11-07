"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with proper 1/5 success rule, multi-start, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0  # safety margin before 330s kill

        def elapsed():
            return time.time() - start_time

        def time_ok():
            return time.time() < deadline

        # Use a fixed seed for reproducibility within a single instance
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        # Multi-start strategy: try several starting points
        # First, do a coarse exploration, then refine
        # We'll use a population-based approach for initial exploration, then (1+1)-ES

        # Phase 1: Coarse exploration with a small population
        pop_size = 10
        for i in range(pop_size):
            if not time_ok() or calls_used >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: (1+1)-ES with 1/5 success rule from the best point
        # Use proper step-size adaptation
        x = best_x.copy()
        fx = best_f
        sigma = 1.0  # Initial step size

        # 1/5 success rule: maintain a window of the last 10 steps
        window_size = 10
        success_history = []
        # Initialize sigma based on the spread of the initial population
        # Actually, let's start with a reasonable sigma
        sigma = 0.5

        while calls_used < self.budget and time_ok():
            # Generate offspring
            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_used += 1

            # Update best if improved
            if fy < best_f:
                best_f = fy
                best_x = y.copy()

            # Track success for step-size adaptation
            success = fy < fx
            success_history.append(success)
            if len(success_history) > window_size:
                success_history.pop(0)

            # Adapt step size using 1/5 success rule
            if len(success_history) == window_size:
                success_rate = sum(success_history) / window_size
                # If success rate > 0.2, increase sigma; else decrease
                if success_rate > 0.2:
                    sigma *= 1.25
                else:
                    sigma *= 0.8

                # Clamp sigma to reasonable bounds
                sigma = np.clip(sigma, 1e-8, 10.0)

            # Update current point
            if success:
                x = y
                fx = fy

        return best_x, best_f
# EVOLVE-BLOCK-END
