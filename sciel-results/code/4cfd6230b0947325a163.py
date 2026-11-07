"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Direct-search pattern method with binary step-size refinement and sparse restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Build search directions: coordinate axes and all pairwise diagonals (positive and negative)
        dim = self.dim
        directions = []
        for i in range(dim):
            d = np.zeros(dim)
            d[i] = 1.0
            directions.append(d)
        for i in range(dim):
            for j in range(i + 1, dim):
                d_pos = np.zeros(dim)
                d_pos[i] = 1.0
                d_pos[j] = 1.0
                directions.append(d_pos)
                d_neg = np.zeros(dim)
                d_neg[i] = 1.0
                d_neg[j] = -1.0
                directions.append(d_neg)

        directions = np.array(directions)
        # Normalize directions to unit length for consistent step interpretation
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / norms

        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        # Phase 1: Sparse random restarts to find a good starting region
        # Use ~30% of budget for exploration
        explore_budget = max(10, int(0.30 * self.budget))
        remaining = self.budget

        for _ in range(explore_budget):
            if not time_ok() or remaining <= 0:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            remaining -= 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            # Fallback: single evaluation at origin
            x0 = np.zeros(dim)
            fx0 = func(x0)
            calls_made += 1
            remaining -= 1
            best_x = x0
            best_f = fx0

        # Phase 2: Direct search with binary step-size refinement
        # We iterate over directions, doing a binary search on step size to find improvement
        base_step = 1.0
        min_step = 1e-6
        max_step = 10.0

        # Track which directions are still active (showing progress)
        active_dirs = list(range(len(directions)))

        while remaining > 0 and time_ok():
            improved_this_pass = False

            for di in active_dirs[:]:  # iterate over copy since we may remove
                if not time_ok() or remaining <= 0:
                    break

                d = directions[di]

                # Binary search on step size in [min_step, max_step]
                lo = min_step
                hi = max_step
                best_local_x = best_x
                best_local_f = best_f
                found_improvement = False

                # Try a few binary search iterations (limited to conserve budget)
                for _ in range(4):
                    if not time_ok() or remaining <= 0:
                        break
                    mid = (lo + hi) / 2.0
                    x_try = np.clip(best_x + mid * d, -5, 5)
                    fx_try = func(x_try)
                    calls_made += 1
                    remaining -= 1

                    if fx_try < best_local_f - 1e-12:
                        best_local_x = x_try.copy()
                        best_local_f = fx_try
                        found_improvement = True
                        lo = mid  # try larger step
                    else:
                        hi = mid  # try smaller step

                if found_improvement:
                    best_x = best_local_x
                    best_f = best_local_f
                    improved_this_pass = True
                else:
                    # No improvement found; try the opposite direction
                    if remaining > 0 and time_ok():
                        x_try = np.clip(best_x - min_step * d, -5, 5)
                        fx_try = func(x_try)
                        calls_made += 1
                        remaining -= 1
                        if fx_try < best_f - 1e-12:
                            best_x = x_try.copy()
                            best_f = fx_try
                            improved_this_pass = True
                        else:
                            # Remove this direction if it's not helping
                            if di in active_dirs:
                                active_dirs.remove(di)

            # If no direction improved, try a small random perturbation (restart-like)
            if not improved_this_pass:
                if not time_ok() or remaining <= 0:
                    break
                # Small random kick
                kick = rng.normal(0, 0.5, dim)
                x_try = np.clip(best_x + kick, -5, 5)
                fx_try = func(x_try)
                calls_made += 1
                remaining -= 1
                if fx_try < best_f - 1e-12:
                    best_x = x_try.copy()
                    best_f = fx_try
                else:
                    # Larger random restart
                    if remaining > 0 and time_ok():
                        x_try = np.clip(best_x + rng.normal(0, 2.0, dim), -5, 5)
                        fx_try = func(x_try)
                        calls_made += 1
                        remaining -= 1
                        if fx_try < best_f - 1e-12:
                            best_x = x_try.copy()
                            best_f = fx_try

            # If all directions removed, do a full random restart
            if not active_dirs:
                if not time_ok() or remaining <= 0:
                    break
                x_try = rng.uniform(-5, 5, dim)
                fx_try = func(x_try)
                calls_made += 1
                remaining -= 1
                if fx_try < best_f - 1e-12:
                    best_x = x_try.copy()
                    best_f = fx_try
                # Reset active directions
                active_dirs = list(range(len(directions)))

        return best_x, best_f
# EVOLVE-BLOCK-END
