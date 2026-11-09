"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-restart CMA-ES with adaptive step size, time-aware budget management.
    Designed for BBOB f1 (Sphere) with hidden shift/rotation/offset.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin before 300s limit

        # Count function calls
        call_count = [0]
        max_calls = self.budget

        def wrapped(x):
            if call_count[0] >= max_calls:
                return np.inf
            call_count[0] += 1
            return func(x)

        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)

        # Phase 1: Multi-start exploration with varying initializations
        # Use a small number of restarts to explore the space
        n_restarts = 3
        calls_per_restart = max(10, (self.budget - n_restarts * 5) // n_restarts)

        for restart in range(n_restarts):
            if call_count[0] >= max_calls:
                break
            if time.time() > deadline - 10:
                break

            # Different initialization strategies
            if restart == 0:
                # Center start
                x0 = np.zeros(self.dim)
            elif restart == 1:
                # Random start
                x0 = rng.uniform(-4, 4, self.dim)
            else:
                # Scaled random start
                x0 = rng.uniform(-3, 3, self.dim)

            fx0 = wrapped(x0)
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

            # CMA-ES-like local optimization from this start
            x = x0.copy()
            fx = fx0
            sigma = 1.5
            success_count = 0
            total_count = 0
            window = 15

            # Adaptive step size CMA-ES
            for i in range(calls_per_restart):
                if call_count[0] >= max_calls:
                    break
                if time.time() > deadline - 5:
                    break

                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = wrapped(y)

                total_count += 1
                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # Rejection: don't move, but still adapt sigma
                    pass

                # Adaptive sigma based on success rate in window
                if total_count % window == 0:
                    success_rate = success_count / window
                    if success_rate > 0.3:
                        sigma *= 1.3
                    elif success_rate < 0.15:
                        sigma *= 0.7
                    else:
                        sigma *= 0.95
                    success_count = 0

                # Final phase: fine-tuning
                if i > calls_per_restart * 0.7:
                    sigma = max(0.01, sigma * 0.9)

        # Phase 2: Final local refinement around best point found
        if best_x is not None and call_count[0] < max_calls:
            if time.time() > deadline - 5:
                pass
            else:
                x = best_x.copy()
                fx = best_f
                sigma = 0.1
                for i in range(min(50, max_calls - call_count[0])):
                    if time.time() > deadline - 2:
                        break
                    y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                    fy = wrapped(y)
                    if fy < fx:
                        x, fx = y, fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    # Decrease sigma for finer search
                    sigma *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
