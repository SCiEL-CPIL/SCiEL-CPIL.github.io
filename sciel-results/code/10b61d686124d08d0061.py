"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Deterministic LHS global search followed by Nelder-Mead local exploitation."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Phase 1: Latin Hypercube Sampling for global basin identification
        # Use a deterministic LHS with a fixed seed for reproducibility
        # Allocate ~30% of budget to global exploration for better basin identification
        n_lhs = max(30, min(budget // 3, 150))
        rng = np.random.default_rng(42)

        # Generate LHS points in [0, 1]^dim
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = rng.permutation(n_lhs)
            lhs_points[:, d] = (perm + 0.5) / n_lhs

        # Map to [-5, 5]^dim
        lhs_points = lhs_points * 10.0 - 5.0

        # Evaluate all LHS points
        best_x = None
        best_f = np.inf
        lhs_results = []
        calls = 0

        for pt in lhs_points:
            pt_clipped = np.clip(pt, -5, 5)
            f_val = func(pt_clipped)
            calls += 1
            lhs_results.append((f_val, pt_clipped))
            if f_val < best_f:
                best_f = f_val
                best_x = pt_clipped.copy()

        # Phase 2: Nelder-Mead simplex method centered on best sample
        # Allocate remaining budget to local optimization
        remaining_budget = budget - calls

        if remaining_budget <= 0:
            return best_x, best_f

        # Initialize Nelder-Mead simplex
        x0 = best_x.copy()
        f0 = best_f

        # Initial simplex size - smaller for faster convergence in smooth regions
        alpha = 0.25  # initial step size for simplex vertices
        simplex = [x0.copy()]
        simplex_f = [f0]

        for d in range(dim):
            vertex = x0.copy()
            vertex[d] += alpha
            vertex = np.clip(vertex, -5, 5)
            f_v = func(vertex)
            calls += 1
            simplex.append(vertex)
            simplex_f.append(f_v)

        if calls >= budget:
            # Find best in simplex
            idx = np.argmin(simplex_f)
            return simplex[idx], simplex_f[idx]

        # Nelder-Mead parameters
        reflection = 1.0
        expansion = 2.0
        contraction = 0.5
        shrink = 0.5

        # Main Nelder-Mead loop
        while calls < budget and len(simplex) == dim + 1:
            # Sort simplex by function value
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]

            # Compute centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_r = centroid + reflection * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls += 1

            if calls >= budget:
                break

            if f_r < simplex_f[0]:
                # Expansion
                x_e = centroid + expansion * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls += 1

                if calls >= budget:
                    break

                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            elif f_r < simplex_f[-2]:
                # Accept reflection
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                # Contraction
                if f_r < simplex_f[-1]:
                    # Outside contraction
                    x_c = centroid + contraction * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - contraction * (x_r - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls += 1

                if calls >= budget:
                    break

                if f_c < min(f_r, simplex_f[-1]):
                    simplex[-1] = x_c
                    simplex_f[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + shrink * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_sh = func(simplex[i])
                        calls += 1
                        simplex_f[i] = f_sh
                        if calls >= budget:
                            break
                    if calls >= budget:
                        break

        # Find best point
        all_points = []
        all_fvals = []

        # Include LHS results
        for f_val, pt in lhs_results:
            all_points.append(pt)
            all_fvals.append(f_val)

        # Include simplex points
        for i in range(len(simplex)):
            all_points.append(simplex[i])
            all_fvals.append(simplex_f[i])

        all_fvals = np.array(all_fvals)
        idx = np.argmin(all_fvals)
        return all_points[idx], all_fvals[idx]
# EVOLVE-BLOCK-END
