"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-stage (1+1)-ES with diverse exploration and adaptive polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        best_x = None
        best_f = float('inf')
        calls_made = 0
        dim = self.dim

        def safe_func(x):
            nonlocal calls_made
            if calls_made >= self.budget:
                return float('inf')
            fx = func(x)
            calls_made += 1
            return fx

        def run_es(x0, fx0, n_calls, sigma0, sigma_min, sigma_max):
            """Run (1+1)-ES for up to n_calls evaluations starting from x0."""
            nonlocal calls_made, best_x, best_f

            x = x0.copy()
            fx = fx0
            sigma = sigma0

            no_improve = 0
            no_improve_limit = max(15, n_calls // 12)

            for i in range(n_calls):
                if calls_made >= self.budget:
                    break
                if time.time() > deadline:
                    break

                step = sigma * np.random.randn(dim)
                y = np.clip(x + step, -5.0, 5.0)

                fy = safe_func(y)
                if fy == float('inf'):
                    break

                if fy < fx:
                    x = y
                    fx = fy
                    sigma = min(sigma * 1.15, sigma_max)
                    no_improve = 0

                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma = max(sigma * 0.85, sigma_min)
                    no_improve += 1

                if no_improve >= no_improve_limit and i < n_calls - 5:
                    break

            return x, fx, sigma

        # Stage 1: Diverse exploration with multiple short ES runs
        # Allocate 40% of budget to exploration
        n_explore = min(int(0.4 * self.budget), self.budget)
        n_starts = 5
        calls_per_start = n_explore // n_starts

        for s in range(n_starts):
            if calls_made >= self.budget or time.time() > deadline:
                break

            if s == 0 or best_x is None:
                x0 = np.random.uniform(-4.0, 4.0, dim)
            else:
                # Mix of random starts and perturbations of best
                if s % 2 == 0 and best_x is not None:
                    x0 = np.clip(best_x + np.random.randn(dim) * 1.5, -5.0, 5.0)
                else:
                    x0 = np.random.uniform(-4.0, 4.0, dim)

            fx0 = safe_func(x0)
            if fx0 == float('inf'):
                break
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

            if calls_made >= self.budget or time.time() > deadline:
                break

            x, fx, sigma = run_es(
                x0, fx0,
                calls_per_start - 1,
                2.0, 0.2, 4.0
            )

        # Stage 2: Main refinement from best point
        if best_x is not None and calls_made < self.budget and time.time() < deadline:
            remaining = self.budget - calls_made
            # Use moderate initial step, allow it to grow if stuck
            x0 = best_x.copy()
            fx0 = best_f
            run_es(x0, fx0, remaining, 0.8, 0.005, 3.0)

        # Stage 3: Final polish with very small steps
        if best_x is not None and calls_made < self.budget and time.time() < deadline:
            remaining = self.budget - calls_made
            if remaining > 30:
                x0 = best_x.copy()
                fx0 = best_f
                run_es(x0, fx0, remaining, 0.05, 0.0001, 0.3)

        return best_x, best_f
# EVOLVE-BLOCK-END
