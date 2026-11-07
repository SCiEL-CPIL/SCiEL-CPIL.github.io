"""Black-box continuous optimizer on BBOB-style f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive step size and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Global exploration - evaluate 50 random points
        n_global = min(50, budget)
        candidates = []
        for i in range(n_global):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            candidates.append((fx, x))
            if time.time() > deadline:
                break

        # Sort by function value and take top 5
        candidates.sort(key=lambda t: t[0])
        top_k = min(5, len(candidates))
        starts = [candidates[i][1].copy() for i in range(top_k)]
        best_fx = candidates[0][0]
        best_x = candidates[0][1].copy()

        calls_made = len(candidates)
        remaining = budget - calls_made

        if remaining <= 0:
            return best_x, best_fx

        # Phase 2: Run (1+1)-ES from each of the top 5 starting points
        # Allocate remaining budget evenly
        n_starts = top_k
        budget_per_start = max(1, remaining // n_starts)

        for si in range(n_starts):
            x = starts[si].copy()
            fx = func(x) if calls_made < budget else candidates[si][0]
            # Actually we already evaluated these, so use the stored value
            fx = candidates[si][0]
            calls_made += 1  # count the initial evaluation (already done)

            sigma = 1.0
            success_count = 0
            attempts = 0

            for step in range(budget_per_start):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    return best_x, best_fx

                # Generate offspring
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    sigma *= 1.25
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85

                # Adaptive sigma adjustment every 10 steps
                attempts += 1
                if attempts % 10 == 0:
                    rate = success_count / attempts
                    if rate > 0.5:
                        sigma *= 1.1
                    elif rate < 0.1:
                        sigma *= 0.9
                    success_count = 0
                    attempts = 0

                # Safety check
                if time.time() > deadline:
                    return best_x, best_fx

        return best_x, best_fx
# EVOLVE-BLOCK-END
