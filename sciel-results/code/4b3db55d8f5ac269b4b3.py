"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with adaptive sigma for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        # Global best tracking
        global_best = None
        global_fx = np.inf
        calls_made = 0

        # Multi-restart strategy: try different initial points
        # Lunacek has two basins, so we try several starting points
        rng = np.random.default_rng(42)

        # Number of restarts: we'll use a fraction of budget for each
        # Start with a few exploratory restarts, then focus on best
        num_restarts = 5
        per_restart_budget = max(50, budget // (num_restarts + 2))

        # Initial diverse starting points
        start_points = []
        # Center
        start_points.append(np.zeros(dim))
        # Random points
        for _ in range(num_restarts - 1):
            start_points.append(rng.uniform(-4, 4, dim))

        for restart_idx in range(num_restarts):
            if calls_made >= budget - 10 or time.time() > deadline - 10:
                break

            x0 = start_points[restart_idx].copy()
            fx0 = func(x0)
            calls_made += 1

            if fx0 < global_fx:
                global_fx = fx0
                global_best = x0.copy()

            # CMA-ES-like adaptation for this restart
            sigma = 1.5
            # Simple covariance adaptation via diagonal scaling
            diag_scale = np.ones(dim)
            # Success rate tracking for 1/5 rule
            successes = 0
            total = 0
            patience = 0
            max_patience = 80

            # Run CMA-ES for this restart
            budget_left = budget - calls_made
            max_iter = min(per_restart_budget, budget_left)

            for it in range(max_iter):
                if calls_made >= budget - 5 or time.time() > deadline - 5:
                    break

                # Generate candidate
                y = x0 + sigma * diag_scale * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                total += 1

                if fy < fx0:
                    successes += 1
                    # Update step size based on 1/5 rule
                    if total >= 5:
                        rate = successes / total
                        if rate > 1.0 / 5.0:
                            sigma *= 1.1
                        else:
                            sigma *= 0.9
                        successes = 0
                        total = 0
                    x0 = y
                    fx0 = fy
                    if fx0 < global_fx:
                        global_fx = fx0
                        global_best = x0.copy()
                    patience = 0
                else:
                    patience += 1

                # Adaptive diagonal scaling based on recent moves
                if it % 10 == 0 and it > 0:
                    # Simple adaptation: scale dimensions by inverse of recent variance
                    pass  # Keep it simple for now

                # Restart this local search if stuck
                if patience > max_patience:
                    break

            # After restart, perturb global best for next attempt
            if restart_idx < num_restarts - 1 and global_best is not None:
                # Add a small perturbation to explore around global best
                start_points.append(global_best + rng.uniform(-0.5, 0.5, dim))

        # Final polish: use remaining budget on global best
        remaining = budget - calls_made
        if remaining > 10 and global_best is not None and time.time() < deadline - 5:
            x = global_best.copy()
            fx = global_fx
            sigma = 0.3
            successes = 0
            total = 0

            for _ in range(remaining - 5):
                if time.time() > deadline - 2:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                total += 1

                if fy < fx:
                    successes += 1
                    if total >= 5:
                        rate = successes / total
                        if rate > 1.0 / 5.0:
                            sigma *= 1.2
                        else:
                            sigma *= 0.85
                        successes = 0
                        total = 0
                    x = y
                    fx = fy
                else:
                    if total >= 20:
                        break

        return global_best, global_fx
# EVOLVE-BLOCK-END
