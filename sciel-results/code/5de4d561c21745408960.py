"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with 1/5 success rule for rotated Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        deadline = time.time() + 280  # Safety margin before 300s

        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            call_count += 1
            try:
                val = func(np.asarray(x, dtype=float).reshape(dim))
                if np.isfinite(val):
                    return float(val)
                return 1e10
            except Exception:
                return 1e10

        def run_es_15_rule(start_x, max_calls, initial_sigma=1.5):
            """(1+1)-ES with 1/5 success rule for sigma adaptation."""
            nonlocal best_x, best_f
            x = np.clip(start_x, -5, 5).copy()
            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = initial_sigma
            used = 1
            success_count = 0
            window_size = 10
            success_window = 0
            stagnation = 0
            max_stagnation = 50

            while used < max_calls and call_count < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_func(y)
                used += 1

                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    success_window += 1
                    stagnation = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    success_window += 1
                    stagnation += 1

                # 1/5 success rule: adjust sigma based on success rate in window
                if success_window >= window_size:
                    success_rate = success_count / success_window
                    if success_rate > 0.2:
                        sigma *= 1.2
                    elif success_rate < 0.2:
                        sigma *= 0.85
                    else:
                        sigma *= 1.0
                    success_count = 0
                    success_window = 0

                # Safety: keep sigma in reasonable bounds
                sigma = np.clip(sigma, 0.01, 5.0)

                # Stagnation check: if no improvement for max_stagnation steps, break
                if stagnation >= max_stagnation:
                    break

            return x, fx

        # Phase 1: Global exploration with multiple diverse restarts
        explore_budget = int(budget * 0.55)
        n_restarts = 6
        per_restart = explore_budget // n_restarts

        for r in range(n_restarts):
            if time.time() > deadline or call_count >= budget:
                break

            # Diverse starting points to cover the search space
            if r == 0:
                x = np.zeros(dim)
            elif r == 1:
                x = rng.uniform(-4, 4, dim)
            elif r == 2:
                x = rng.uniform(-3, 3, dim)
            elif r == 3:
                x = rng.uniform(-5, 5, dim)
            elif r == 4:
                x = rng.uniform(-2, 2, dim)
            else:
                x = rng.uniform(-4.5, 4.5, dim)

            x = np.clip(x, -5, 5)
            run_es_15_rule(x, per_restart, initial_sigma=2.0)

        # Phase 2: Intensive exploitation around best point
        if best_x is not None and call_count < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            stagnation = 0
            max_stagnation = 30
            escape_attempts = 0

            while call_count < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_func(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.15, 2.0)
                    stagnation = 0
                    escape_attempts = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
                    stagnation += 1

                if stagnation > max_stagnation:
                    # Escape mechanism: try random perturbations to find new basin
                    for _ in range(5):
                        if call_count >= budget or time.time() > deadline:
                            break
                        y = np.clip(x + 1.5 * rng.standard_normal(dim), -5, 5)
                        fy = safe_func(y)
                        if fy < fx:
                            x, fx = y, fy
                            sigma = 1.2
                            stagnation = 0
                            escape_attempts = 0
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            break
                    else:
                        # No escape found, try larger perturbation
                        if escape_attempts < 3:
                            y = np.clip(x + 2.5 * rng.standard_normal(dim), -5, 5)
                            fy = safe_func(y)
                            if fy < fx:
                                x, fx = y, fy
                                sigma = 1.5
                                stagnation = 0
                                escape_attempts = 0
                                if fx < best_f:
                                    best_f = fx
                                    best_x = x.copy()
                            else:
                                escape_attempts += 1
                                stagnation = 0  # Reset to continue searching
                        else:
                            # Give up on this basin, stop
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
