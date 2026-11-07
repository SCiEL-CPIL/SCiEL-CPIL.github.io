"""Optimizer for BBOB f24 (Lunacek bi-Rastrigin) with multi-start and momentum."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with momentum and adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit, keep 20s margin
        rng = np.random.default_rng(42)
        n = self.dim
        total_budget = self.budget
        calls = 0

        best_x = None
        best_f = np.inf

        def _eval(x):
            nonlocal calls
            calls += 1
            return func(x)

        # Phase 1: Multi-start with diverse initial points
        # Lunacek has two basins; try to find both
        n_starts = min(8, total_budget // 50)
        start_budget = min(50, total_budget // n_starts)

        starts = []
        # First start: random
        x0 = rng.uniform(-5, 5, n)
        starts.append(x0)

        # Additional starts spread across the domain
        for i in range(1, n_starts):
            if i % 2 == 1:
                x0 = rng.uniform(-4, 0, n)
            else:
                x0 = rng.uniform(0, 4, n)
            starts.append(x0)

        # Run (1+1)-ES with momentum for each start
        for start_idx, x0 in enumerate(starts):
            if calls >= total_budget:
                break
            if time.time() > deadline:
                break

            x = np.array(x0, dtype=float)
            fx = _eval(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.5
            momentum = np.zeros(n)
            beta = 0.3  # momentum coefficient
            no_improve = 0
            max_no_improve = 15

            remaining = total_budget - calls
            if remaining <= 0:
                break

            for step in range(min(start_budget, remaining)):
                if time.time() > deadline:
                    break

                # Generate candidate with momentum
                direction = rng.standard_normal(n)
                if no_improve > 5:
                    # Increase exploration
                    direction = direction + 0.5 * rng.standard_normal(n)
                    direction /= np.linalg.norm(direction) + 1e-12

                # Update momentum
                momentum = beta * momentum + (1 - beta) * direction

                # Propose new point
                y = x + sigma * momentum
                y = np.clip(y, -5, 5)
                fy = _eval(y)

                if fy < fx:
                    x = y
                    fx = fy
                    sigma = min(sigma * 1.2, 5.0)
                    no_improve = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma = max(sigma * 0.85, 0.05)
                    no_improve += 1

                if no_improve >= max_no_improve:
                    # Restart from current best with perturbation
                    x = best_x + 0.5 * rng.standard_normal(n)
                    x = np.clip(x, -5, 5)
                    fx = _eval(x)
                    sigma = 2.0
                    momentum = np.zeros(n)
                    no_improve = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    break

        # Phase 2: Fine-tuning around the best point
        if best_x is not None and calls < total_budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
            momentum = np.zeros(n)
            beta = 0.2
            no_improve = 0
            remaining = total_budget - calls

            for step in range(remaining):
                if time.time() > deadline:
                    break

                direction = rng.standard_normal(n)
                momentum = beta * momentum + (1 - beta) * direction

                y = x + sigma * momentum
                y = np.clip(y, -5, 5)
                fy = _eval(y)

                if fy < fx:
                    x = y
                    fx = fy
                    sigma = min(sigma * 1.1, 2.0)
                    no_improve = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma = max(sigma * 0.9, 0.01)
                    no_improve += 1

                    if no_improve >= 20:
                        # Try a larger jump to escape local minimum
                        jump = 1.0 * rng.standard_normal(n)
                        y2 = np.clip(x + jump, -5, 5)
                        fy2 = _eval(y2)
                        if fy2 < fx:
                            x = y2
                            fx = fy2
                            sigma = 0.5
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                        no_improve = 0

        if best_x is None:
            # Fallback: should not happen, but just in case
            best_x = np.zeros(n)
            best_f = _eval(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
