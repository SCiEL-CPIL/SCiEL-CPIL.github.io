"""Rewritten optimizer: multi-restart (1+1)-ES with sigma adaptation and time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with 1/5 success-rule sigma adaptation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def remaining_time():
            return deadline - time.time()

        def is_timeup():
            return time.time() >= deadline

        # We'll do multiple restarts. Each restart runs a (1+1)-ES.
        # Allocate budget across restarts: first restart gets more, later ones less.
        # Strategy: split budget into chunks, decreasing.

        # Determine number of restarts based on budget
        # For budget=300, dim=5: do about 5-10 restarts
        n_restarts = min(10, max(3, budget // 30))

        # Budget allocation: give first restart 40%, then split rest among others
        remaining_budget = budget
        budgets = []
        for i in range(n_restarts):
            if i == 0:
                b = int(remaining_budget * 0.4)
            else:
                b = int(remaining_budget / (n_restarts - i))
            budgets.append(b)
            remaining_budget -= b
        # Ensure total doesn't exceed budget
        total_alloc = sum(budgets)
        if total_alloc > budget:
            budgets[-1] -= (total_alloc - budget)

        # Starting points for restarts
        start_points = []
        # First start: random
        start_points.append(rng.uniform(-5, 5, self.dim))
        # Subsequent starts: random, but also try around best found
        for i in range(1, n_restarts):
            if best_x is not None and i % 2 == 0:
                # Perturb best
                pert = best_x + 1.5 * rng.standard_normal(self.dim)
                pert = np.clip(pert, -5, 5)
            else:
                pert = rng.uniform(-5, 5, self.dim)
            start_points.append(pert)

        for restart_idx in range(n_restarts):
            if calls_used >= budget or is_timeup():
                break

            x = start_points[restart_idx].copy()
            fx = func(x)
            calls_used += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 2.0
            # Adapt sigma using 1/5 success rule
            success_count = 0
            attempts = 0
            max_attempts_for_sigma = 10  # check every 10 steps

            alloc = budgets[restart_idx] if restart_idx < len(budgets) else max(1, budget // n_restarts)

            for step in range(alloc):
                if calls_used >= budget or is_timeup():
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_used += 1

                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                attempts += 1

                if attempts >= max_attempts_for_sigma:
                    # 1/5 success rule
                    if success_count / attempts > 0.2:
                        sigma *= 1.3  # slightly faster growth
                    else:
                        sigma *= 0.85  # slower decay
                    sigma = np.clip(sigma, 0.01, 10.0)
                    success_count = 0
                    attempts = 0

            # After this restart, if we found a good solution, subsequent
            # restarts will be seeded near it (handled in start_points generation
            # for even indices, but let's also update start_points for odd ones)
            if restart_idx < len(start_points) - 1:
                # Update next start point to be near best
                if restart_idx + 1 < len(start_points):
                    if (restart_idx + 1) % 2 == 0:
                        pert = best_x + 1.5 * rng.standard_normal(self.dim)
                        start_points[restart_idx + 1] = np.clip(pert, -5, 5)

        if best_x is None:
            # Fallback: single evaluation
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            best_x = x
            best_f = fx

        return best_x, best_f
# EVOLVE-BLOCK-END
