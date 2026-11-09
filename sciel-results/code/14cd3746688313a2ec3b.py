"""Deterministic Nelder-Mead simplex optimizer for BBOB f18."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead simplex method with adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Nelder-Mead parameters
        alpha = 1.0    # reflection
        gamma = 2.0    # expansion
        rho = 0.5      # contraction
        sigma = 0.5    # shrink
        min_sigma = 1e-12

        # Track best overall
        best_x = None
        best_f = np.inf
        calls = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Initial simplex at centroid of the box
        center = np.zeros(dim)

        # Initial simplex vertices
        simplex = []
        fvals = []

        # First vertex at center
        v0 = center.copy()
        simplex.append(v0)
        f0 = func(v0)
        calls += 1
        fvals.append(f0)
        update_best(v0, f0)

        # Additional vertices offset along each axis
        for i in range(dim):
            vi = center.copy()
            vi[i] = 1.0  # small initial step
            simplex.append(vi)
            fi = func(vi)
            calls += 1
            fvals.append(fi)
            update_best(vi, fi)

        # Main optimization loop
        iteration = 0
        while calls < budget:
            # Sort simplex by function values
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Check for convergence
            f_spread = max(fvals) - min(fvals)
            if f_spread < 1e-12 and max(np.abs(np.array(fvals) - fvals[0])) < 1e-12:
                break

            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]

            # Reflection
            xr = centroid + alpha * (centroid - worst)
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls += 1
            update_best(xr, fr)

            if fr < fvals[0]:
                # Try expansion
                xe = centroid + gamma * (centroid - worst)
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls += 1
                update_best(xe, fe)

                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                # Accept reflection
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - worst)

                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls += 1
                update_best(xc, fc)

                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fi = func(simplex[i])
                        fvals[i] = fi
                        calls += 1
                        update_best(simplex[i], fi)
                        if calls >= budget:
                            break
                    if calls >= budget:
                        break

            iteration += 1

            # If simplex has collapsed, restart with a new random-ish point
            # near the current best
            if iteration % 50 == 0:
                # Check if simplex is too small
                spread = np.max(np.abs(np.array(simplex) - best_x))
                if spread < 1e-6:
                    # Restart around best point with a small perturbation
                    simplex = [best_x.copy()]
                    fvals = [best_f]
                    for i in range(dim):
                        vi = best_x.copy()
                        vi[i] += 0.5
                        vi = np.clip(vi, -5, 5)
                        simplex.append(vi)
                        fi = func(vi)
                        fvals.append(fi)
                        calls += 1
                        update_best(vi, fi)
                        if calls >= budget:
                            break
                    if calls >= budget:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
