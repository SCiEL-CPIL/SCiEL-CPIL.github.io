"""Nelder-Mead simplex with adaptive restarts for Rastrigin f3 in 5D."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex optimizer with adaptive restarts for periodic Rastrigin landscape."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280
        rng = np.random.default_rng(12345)

        calls = [0]

        def call_func(x):
            calls[0] += 1
            return float(func(x))

        def time_ok():
            return time.time() < deadline and calls[0] < budget

        # ---------- Phase 1: Initial exploration using integer lattice sampling ----------
        # For Rastrigin, local minima are at integer offsets from the shift.
        # Sample a grid of integer points to find promising basins.
        best_x = None
        best_f = np.inf

        # Sample integer points in [-4, 4]^5 with some randomization
        # Use a stratified approach: sample from a coarser grid
        n_grid = 3  # 3 points per axis in [-4, 4] -> {-3, 0, 3} or similar
        grid_vals = np.linspace(-4, 4, n_grid)

        # Also include 0 and nearby
        candidates = []
        for v1 in grid_vals:
            for v2 in grid_vals:
                for v3 in grid_vals:
                    for v4 in grid_vals:
                        for v5 in grid_vals:
                            candidates.append(np.array([v1, v2, v3, v4, v5]))

        # Add origin and unit vectors
        candidates.append(np.zeros(dim))
        for axis in range(dim):
            for delta in [-1, 1]:
                pt = np.zeros(dim)
                pt[axis] = delta
                candidates.append(pt)

        # Shuffle and evaluate up to 40 points
        rng.shuffle(candidates)
        n_explore = min(40, len(candidates))
        for i in range(n_explore):
            if not time_ok():
                break
            x_cand = np.clip(candidates[i], -5, 5)
            fx = call_func(x_cand)
            if fx < best_f:
                best_f = fx
                best_x = x_cand.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = call_func(best_x)

        # ---------- Phase 2: Nelder-Mead with adaptive restarts ----------
        def nelder_mead(x0, f0, max_iter, initial_alpha=0.5):
            """Run Nelder-Mead simplex optimization from x0."""
            dim_local = dim
            alpha = initial_alpha  # step size / simplex scale

            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [f0]

            # Add perturbed points
            for i in range(dim_local):
                x_new = x0.copy()
                x_new[i] += alpha
                x_new = np.clip(x_new, -5, 5)
                f_new = call_func(x_new)
                simplex.append(x_new)
                fvals.append(f_new)

            # Sort
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            best_x_local = simplex[0].copy()
            best_f_local = fvals[0]

            # Nelder-Mead parameters
            gamma = 1.0   # reflection
            rho = 2.0     # expansion
            sigma = 0.5   # contraction
            shrink = 0.5  # shrink

            for it in range(max_iter):
                if not time_ok():
                    break

                # Check if we've converged
                if fvals[-1] - fvals[0] < 1e-10:
                    break

                # Reflection
                centroid = np.mean(simplex[:-1], axis=0)
                x_reflect = centroid + gamma * (centroid - simplex[-1])
                x_reflect = np.clip(x_reflect, -5, 5)
                f_reflect = call_func(x_reflect)

                if f_reflect < fvals[0]:
                    # Expansion
                    x_expand = centroid + rho * (x_reflect - centroid)
                    x_expand = np.clip(x_expand, -5, 5)
                    f_expand = call_func(x_expand)
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        fvals[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        fvals[-1] = f_reflect
                elif f_reflect < fvals[-2]:
                    simplex[-1] = x_reflect
                    fvals[-1] = f_reflect
                else:
                    # Contraction
                    if f_reflect < fvals[-1]:
                        x_contract = centroid + sigma * (x_reflect - centroid)
                        f_contract = call_func(x_contract)
                    else:
                        x_contract = centroid - sigma * (simplex[-1] - centroid)
                        f_contract = call_func(x_contract)

                    x_contract = np.clip(x_contract, -5, 5)
                    if f_contract < f_reflect:
                        simplex[-1] = x_contract
                        fvals[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, dim_local + 1):
                            simplex[i] = simplex[0] + shrink * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = call_func(simplex[i])

                # Re-sort
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Update best
                if fvals[0] < best_f_local:
                    best_f_local = fvals[0]
                    best_x_local = simplex[0].copy()

                # Adaptive: if simplex is too small, expand it
                spread = np.max(np.array(simplex), axis=0) - np.min(np.array(simplex), axis=0)
                if np.max(spread) < 0.01:
                    # Restart with larger step
                    for i in range(1, dim_local + 1):
                        x_new = best_x_local + alpha * 2 * rng.standard_normal(dim_local)
                        x_new = np.clip(x_new, -5, 5)
                        simplex[i] = x_new
                        fvals[i] = call_func(x_new)
                    alpha *= 1.5
                    order = np.argsort(fvals)
                    simplex = [simplex[i] for i in order]
                    fvals = [fvals[i] for i in order]
                    if fvals[0] < best_f_local:
                        best_f_local = fvals[0]
                        best_x_local = simplex[0].copy()

            return best_x_local, best_f_local

        # Run Nelder-Mead with multiple restarts
        remaining = budget - calls[0]
        n_restarts = 3
        budget_per_restart = remaining // n_restarts

        for restart in range(n_restarts):
            if not time_ok():
                break

            if restart == 0:
                # First run from best found
                x_start = best_x.copy()
                f_start = best_f
                alpha_init = 0.5
            else:
                # Subsequent runs: perturb from best
                x_start = best_x + 0.3 * rng.standard_normal(dim)
                x_start = np.clip(x_start, -5, 5)
                f_start = call_func(x_start)
                if f_start < best_f:
                    best_f = f_start
                    best_x = x_start.copy()
                alpha_init = 0.3

            x_result, f_result = nelder_mead(x_start, f_start, budget_per_restart, alpha_init)

            if f_result < best_f:
                best_f = f_result
                best_x = x_result.copy()

        # ---------- Phase 3: Final fine-tuning with small steps ----------
        if time_ok():
            sigma = 0.05
            for _ in range(budget - calls[0]):
                if not time_ok():
                    break
                step = sigma * rng.standard_normal(dim)
                y = np.clip(best_x + step, -5, 5)
                fy = call_func(y)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                if sigma < 0.001:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
