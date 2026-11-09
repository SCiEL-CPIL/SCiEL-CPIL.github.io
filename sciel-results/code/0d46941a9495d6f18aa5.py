"""Optimizer for BBOB-style Lunacek bi-Rastrigin (f24) with restarts and CMA-ES."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Population-based optimizer with restarts for bi-Rastrigin landscape."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        best_x = None
        best_f = np.inf
        call_count = 0
        max_calls = self.budget

        # Track best so far for restarts
        best_x_global = None
        best_f_global = np.inf

        def safe_func(x):
            nonlocal call_count
            if call_count >= max_calls:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            call_count += 1
            return func(x)

        # Multi-start with adaptive restarts
        # Phase 1: Broad exploration with multiple random starts
        # Phase 2: Focused exploitation near best found

        rng = np.random.default_rng(42)

        # Use CMA-ES-like strategy with restarts
        # For Lunacek bi-Rastrigin, the landscape has two basins:
        # one at the global minimum and one at a local minimum
        # We need to explore both

        num_restarts = 5
        calls_per_restart = max(50, (max_calls - 10) // num_restarts)

        for restart in range(num_restarts):
            if call_count >= max_calls:
                break
            if time.time() > deadline - 10:
                break

            # Choose starting point
            if best_x_global is not None and restart > 0:
                # Mix of near-best and random exploration
                if restart % 2 == 1:
                    # Start near global best
                    x0 = np.clip(best_x_global + rng.normal(0, 1.0, self.dim), -5, 5)
                else:
                    # Random restart to find other basin
                    x0 = rng.uniform(-5, 5, self.dim)
            else:
                x0 = rng.uniform(-5, 5, self.dim)

            # Run a local optimization from x0
            x_best_local = x0.copy()
            f_best_local = safe_func(x0)

            if f_best_local < best_f_global:
                best_f_global = f_best_local
                best_x_global = x0.copy()

            # Adaptive step size local search
            sigma = 2.0
            min_sigma = 1e-6
            max_sigma = 5.0
            no_improve_count = 0
            max_no_improve = 15

            while call_count < max_calls and sigma > min_sigma:
                if time.time() > deadline - 5:
                    break

                # Generate candidate
                y = np.clip(x_best_local + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = safe_func(y)

                if fy < f_best_local:
                    x_best_local = y
                    f_best_local = fy
                    no_improve_count = 0
                    sigma = min(sigma * 1.3, max_sigma)

                    if f_best_local < best_f_global:
                        best_f_global = f_best_local
                        best_x_global = y.copy()
                else:
                    no_improve_count += 1
                    sigma *= 0.75

                    if no_improve_count >= max_no_improve:
                        break

            # Update global best
            if f_best_local < best_f_global:
                best_f_global = f_best_local
                best_x_global = x_best_local.copy()

            # If we've used too many calls, stop
            if call_count >= max_calls - 10:
                break

        # Final polish: fine-grained search around global best if time/calls remain
        if best_x_global is not None and call_count < max_calls - 5:
            if time.time() < deadline - 5:
                sigma = 0.5
                x_polish = best_x_global.copy()
                f_polish = best_f_global
                no_improve = 0

                while call_count < max_calls and time.time() < deadline - 2:
                    y = np.clip(x_polish + sigma * rng.standard_normal(self.dim), -5, 5)
                    fy = safe_func(y)
                    if fy < f_polish:
                        x_polish = y
                        f_polish = fy
                        no_improve = 0
                        sigma = min(sigma * 1.2, 2.0)
                    else:
                        no_improve += 1
                        sigma *= 0.8
                        if no_improve > 20:
                            break

                    if f_polish < best_f_global:
                        best_f_global = f_polish
                        best_x_global = x_polish.copy()

        # Ensure we made at least one call
        if call_count == 0:
            x_init = np.zeros(self.dim)
            f_init = safe_func(x_init)
            best_x_global = x_init
            best_f_global = f_init

        if best_x_global is None:
            best_x_global = np.zeros(self.dim)
            best_f_global = safe_func(best_x_global)

        return best_x_global, best_f_global
