"""Rewritten program: LHS global search + Nelder-Mead local exploitation."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Dense Latin Hypercube sampling for basin identification followed by
    fixed-step Nelder-Mead simplex local exploitation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Allocate budget: ~60% for global LHS, ~40% for local NM
        n_lhs = max(10, int(0.6 * budget))
        n_nm = budget - n_lhs

        # --- Phase 1: Dense Latin Hypercube Sampling ---
        # Generate LHS points in [0,1]^dim, then scale to [-5,5]^dim
        # For robustness, use multiple LHS samples if budget allows, but stick to one dense set
        n_samples = n_lhs
        # Standard LHS: for each dimension, partition [0,1] into n_samples intervals,
        # pick one point uniformly in each, then permute across dimensions
        lhs_points = np.empty((n_samples, dim))
        for d in range(dim):
            # Uniform points in each of n_samples bins
            pts = (np.arange(n_samples) + rng.uniform(size=n_samples)) / n_samples
            # Random permutation to decorrelate dimensions
            perm = rng.permutation(n_samples)
            lhs_points[:, d] = pts[perm]
        
        # Scale to [-5, 5]
        lhs_points = -5.0 + 10.0 * lhs_points

        # Evaluate all LHS points
        best_x = None
        best_f = np.inf
        for i in range(n_samples):
            x = lhs_points[i]
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # --- Phase 2: Nelder-Mead Simplex Local Search ---
        # Center on the best LHS point
        x0 = best_x.copy()
        f0 = best_f

        # Initialize simplex
        # Standard NM: n+1 points
        simplex = np.empty((dim + 1, dim))
        simplex[0] = x0
        for i in range(1, dim + 1):
            simplex[i] = x0.copy()
            simplex[i, i - 1] += 0.05  # small initial step
            # Clip to bounds
            simplex[i] = np.clip(simplex[i], -5, 5)

        # Evaluate simplex
        fvals = np.empty(dim + 1)
        fvals[0] = f0
        for i in range(1, dim + 1):
            fvals[i] = func(simplex[i])

        # NM parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink
        max_iter = n_nm
        tol = 1e-10

        # Track best overall
        global_best_x = best_x.copy()
        global_best_f = best_f

        for it in range(max_iter):
            # Sort simplex by fitness
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

            # Check convergence
            if np.ptp(fvals) < tol:
                break

            # Centroid of all points except worst
            x_c = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_r = x_c + alpha * (x_c - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)

            if f_r < fvals[0]:
                # Expansion
                x_e = x_c + gamma * (x_r - x_c)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                # Simple reflection
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                if f_r < fvals[-1]:
                    # Outside contraction
                    x_c2 = x_c + rho * (x_r - x_c)
                    x_c2 = np.clip(x_c2, -5, 5)
                    f_c = func(x_c2)
                    if f_c <= f_r:
                        simplex[-1] = x_c2
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                        break
                else:
                    # Inside contraction
                    x_c2 = x_c - rho * (x_c - simplex[-1])
                    x_c2 = np.clip(x_c2, -5, 5)
                    f_c = func(x_c2)
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c2
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                        break

            # Update global best
            if fvals[0] < global_best_f:
                global_best_f = fvals[0]
                global_best_x = simplex[0].copy()

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
