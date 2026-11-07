"""Refined multi-restart (1+1)-ES for BBOB f23 (Katsuura)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive step size and final polish for Katsuura."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(2024)

        best_x = None
        best_f = np.inf
        used_calls = 0

        def check_time():
            return time.time() < deadline

        def try_point(x, budget_left):
            nonlocal best_x, best_f, used_calls
            if used_calls >= budget or not check_time():
                return False
            fx = func(x)
            used_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return True

        # Phase 1: Multi-restart (1+1)-ES for global search
        # Katsuura is highly multimodal, so we use many short restarts.
        # 60% budget for global, 40% for local polish.
        global_budget = int(budget * 0.6)
        local_budget = budget - global_budget

        # Use more, shorter restarts to better cover the search space
        num_restarts = 15
        calls_per_restart = global_budget // num_restarts
        if calls_per_restart < 15:
            num_restarts = max(1, global_budget // 15)
            calls_per_restart = global_budget // num_restarts

        for r in range(num_restarts):
            if used_calls >= global_budget or not check_time():
                break

            # Random start for this restart
            x = rng.uniform(-5, 5, d)
            fx = func(x)
            used_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.0
            # (1+1)-ES loop for this restart
            for _ in range(calls_per_restart):
                if used_calls >= global_budget or not check_time():
                    break

                # Generate offspring
                z = rng.standard_normal(d)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                used_calls += 1

                if fy < fx:
                    # Success: move to offspring
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    # Adaptive sigma: increase on success (conservative)
                    sigma *= 1.03
                else:
                    # Failure: decrease sigma
                    sigma *= 0.80

                # Clamp sigma to reasonable range
                sigma = np.clip(sigma, 1e-5, 10.0)

                # Early termination if sigma is very small (converged locally)
                if sigma < 1e-4:
                    break

        # Phase 2: Local Polish - multiple small random walks with decreasing step
        if best_x is not None and used_calls < budget and check_time():
            x = best_x.copy()
            fx = best_f
            remaining = budget - used_calls

            # Try several step sizes, from coarse to fine
            step_sizes = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
            calls_in_phase = 0
            max_calls_in_phase = remaining

            for ss in step_sizes:
                if calls_in_phase >= max_calls_in_phase or not check_time():
                    break

                # Try a fixed number of steps with this size
                attempts = max(3, max_calls_in_phase // (len(step_sizes) * 2))
                for _ in range(attempts):
                    if calls_in_phase >= max_calls_in_phase or not check_time():
                        break

                    z = rng.standard_normal(d)
                    y = x + ss * z
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_in_phase += 1
                    used_calls += 1

                    if fy < fx:
                        x = y
                        fx = fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()

        # Phase 3: Final high-precision polish with very small steps
        if best_x is not None and used_calls < budget and check_time():
            x = best_x.copy()
            fx = best_f
            remaining = budget - used_calls

            # Very fine-grained search
            for ss in [0.0005, 0.0001]:
                if remaining <= 0 or not check_time():
                    break
                attempts = min(remaining, 10)
                for _ in range(attempts):
                    if remaining <= 0 or not check_time():
                        break
                    z = rng.standard_normal(d)
                    y = x + ss * z
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    remaining -= 1
                    used_calls += 1
                    if fy < fx:
                        x = y
                        fx = fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
