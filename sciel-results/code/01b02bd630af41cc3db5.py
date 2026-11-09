"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio of diverse (1+1)-ES runs with time and budget management."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill
        dim = self.dim
        total_budget = self.budget
        calls_used = 0

        best_x = None
        best_f = np.inf

        def track_call(x, f):
            nonlocal calls_used, best_x, best_f
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def check_deadline():
            return time.time() > deadline or calls_used >= total_budget

        def es_run(rng, x0, sigma0, budget_left):
            """(1+1)-ES with 1/5 success rule and adaptive sigma."""
            nonlocal calls_used, best_x, best_f
            x = x0.copy()
            fx = func(x)
            track_call(x, fx)
            sigma = sigma0
            n_dim = self.dim
            success_window = []
            window_size = 10

            for _ in range(budget_left):
                if check_deadline():
                    return

                step = sigma * rng.standard_normal(n_dim)
                y = np.clip(x + step, -5, 5)
                fy = func(y)
                track_call(y, fy)

                if fy < fx:
                    x = y
                    fx = fy
                    success_window.append(1)
                else:
                    success_window.append(0)

                if len(success_window) >= window_size:
                    success_rate = sum(success_window) / window_size
                    if success_rate > 0.2:
                        sigma *= 1.05
                    elif success_rate < 0.2:
                        sigma *= 0.95
                    success_window = success_window[-window_size // 2:]

                sigma = np.clip(sigma, 1e-7, 8.0)

        def polish_run(rng, x0, budget_left):
            """Final high-precision polish with very small steps."""
            nonlocal calls_used, best_x, best_f
            x = x0.copy()
            fx = func(x)
            track_call(x, fx)
            sigma = 0.005
            n_dim = self.dim

            for _ in range(budget_left):
                if check_deadline():
                    return

                step = sigma * rng.standard_normal(n_dim)
                y = np.clip(x + step, -5, 5)
                fy = func(y)
                track_call(y, fy)

                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.02
                else:
                    sigma *= 0.97

                sigma = np.clip(sigma, 1e-9, 0.05)

        # Strategy: Multiple restarts with different initializations and sigma
        # 1. Start with a broad random search to find a good region
        # 2. Then refine with (1+1)-ES
        # 3. Final polish pass for high-precision convergence

        # Phase 1: Broad exploration (8% of budget)
        explore_budget = max(8, total_budget // 12)
        rng = np.random.default_rng(42)

        for _ in range(explore_budget):
            if check_deadline():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            track_call(x, fx)

        # Phase 2: Refinement with multiple ES runs
        remaining_budget = total_budget - calls_used

        if best_x is None:
            # Fallback if no calls were made (shouldn't happen)
            best_x = np.zeros(dim)
            best_f = func(best_x)
            track_call(best_x, best_f)
            remaining_budget = total_budget - calls_used

        if best_x is not None and remaining_budget > 10:
            # Run 1: Refine from the best found point with medium sigma
            es_run(rng, best_x, 1.0, min(remaining_budget, total_budget // 3))
            remaining_budget = total_budget - calls_used

        if remaining_budget > 10:
            # Run 2: New random start with larger sigma for exploration
            x2 = rng.uniform(-5, 5, dim)
            es_run(rng, x2, 2.5, min(remaining_budget, total_budget // 4))
            remaining_budget = total_budget - calls_used

        if remaining_budget > 10:
            # Run 3: New random start with small sigma for fine-tuning
            x3 = rng.uniform(-5, 5, dim)
            es_run(rng, x3, 0.5, min(remaining_budget, total_budget // 4))
            remaining_budget = total_budget - calls_used

        if remaining_budget > 10:
            # Run 4: Another random start
            x4 = rng.uniform(-5, 5, dim)
            es_run(rng, x4, 1.5, min(remaining_budget, total_budget // 5))
            remaining_budget = total_budget - calls_used

        # Phase 3: Final high-precision polish
        if best_x is not None and remaining_budget > 5:
            polish_run(rng, best_x, remaining_budget)

        # Ensure we return something
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            track_call(best_x, best_f)

        return best_x, best_f
# EVOLVE-BLOCK-END
