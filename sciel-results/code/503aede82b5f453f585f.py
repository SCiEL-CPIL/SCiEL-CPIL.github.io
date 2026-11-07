"""Rewritten optimizer: direction-adaptive search (Powell-like) + (1+1)-ES refinement with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Direction-adaptive search followed by (1+1)-ES refinement on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s
        dim = self.dim
        budget = self.budget

        # Track number of function calls
        call_count = [0]

        def f(x):
            call_count[0] += 1
            return func(x)

        # Initial point
        rng = np.random.default_rng(42)
        x = np.zeros(dim)
        fx = f(x)
        best_x = x.copy()
        best_fx = fx

        # Initial directions: coordinate axes
        directions = [np.eye(dim, dtype=float).copy()[i] for i in range(dim)]

        def time_ok():
            return time.time() < deadline and call_count[0] < budget

        def line_search(x, d, f, fx, call_count, time_ok, dim):
            """Find best point along direction d from x."""
            # Try steps in both directions
            best_x_ls = x.copy()
            best_fx_ls = fx
            step = 1.0
            for _ in range(20):
                if not time_ok() or step < 1e-12:
                    break
                for sign in (1, -1):
                    y = np.clip(x + sign * step * d, -5, 5)
                    fy = f(y)
                    if fy < best_fx_ls:
                        best_fx_ls = fy
                        best_x_ls = y.copy()
                step *= 0.7
            return best_x_ls, best_fx_ls

        # Powell-like phase: use ~400 calls for direction adaptation
        powell_budget = min(400, budget)
        max_powell_iters = 15

        for it in range(max_powell_iters):
            if not time_ok() or call_count[0] >= powell_budget:
                break

            x_old = x.copy()
            fx_old = fx

            # Line search along each direction
            for i in range(dim):
                if not time_ok() or call_count[0] >= powell_budget:
                    break
                x, fx = line_search(x, directions[i], f, fx, call_count, time_ok, dim)

            # Check if significant progress was made
            if abs(fx_old - fx) < 1e-14:
                break

            # Try the new direction (from x_old to x)
            new_dir = x - x_old
            norm = np.linalg.norm(new_dir)
            if norm > 1e-14:
                new_dir /= norm
                x_new_dir, fx_new_dir = line_search(x_old, new_dir, f, fx_old, call_count, time_ok, dim)

                if fx_new_dir < fx:
                    # Replace the direction that took longest (here, replace the worst one)
                    # Find the direction with the largest line search cost (approximate: replace last)
                    # Simple strategy: replace the direction that was used last in the loop
                    directions[dim - 1] = new_dir
                    x, fx = x_new_dir, fx_new_dir
                    # Actually we should replace the direction that took the most time,
                    # but for simplicity, let's replace the one that gave the least improvement
                    # For now, just replace the last direction
                else:
                    # No improvement along new direction, just keep x, fx
                    pass
            else:
                break

            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # (1+1)-ES refinement phase with remaining budget
        sigma = 0.5
        x = best_x.copy()
        fx = best_fx

        # Adaptive sigma: 1/5 success rule
        successes = 0
        evals_since_adapt = 0
        window = 10

        while time_ok():
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = f(y)
            evals_since_adapt += 1

            if fy < fx:
                x, fx = y, fy
                successes += 1
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                pass

            if evals_since_adapt >= window:
                if successes / window > 0.5:
                    sigma *= 1.5
                else:
                    sigma *= 0.7
                successes = 0
                evals_since_adapt = 0

        return best_x, best_fx
# EVOLVE-BLOCK-END
