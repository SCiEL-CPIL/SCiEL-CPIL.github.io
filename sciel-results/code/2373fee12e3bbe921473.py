"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global sampling + aggressive Nelder-Mead for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Global sampling to find a good basin
        # Rastrigin has many local minima; we need to sample broadly
        n_global = min(100, budget // 5)
        best_x = None
        best_f = np.inf

        # Sample from multiple scales
        for i in range(n_global):
            if time.time() > deadline:
                break
            # Mix of broad and medium-scale sampling
            if i < 50:
                x = rng.uniform(-5, 5, dim)
            else:
                # Sample around origin and some structured points
                x = rng.uniform(-3, 3, dim)

            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Also try some structured points
        structured = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.ones(dim) * 1.0,
            np.ones(dim) * -1.0,
        ]
        for x in structured:
            if time.time() > deadline:
                break
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Nelder-Mead refinement from best point
        if time.time() < deadline and best_x is not None:
            best_x = self._nelder_mead(func, best_x, best_f, dim, deadline, budget, rng)
            best_f = func(best_x)

        # Phase 3: If we have budget left, do more local refinements from random points
        if time.time() < deadline:
            remaining_budget = budget - n_global - 5
            n_local = min(5, max(1, remaining_budget // 50))
            for _ in range(n_local):
                if time.time() > deadline:
                    break
                # Sample a new starting point
                x0 = rng.uniform(-4, 4, dim)
                f0 = func(x0)
                if f0 < best_f * 1.5:  # Only refine if reasonably close
                    x_refined = self._nelder_mead(func, x0, f0, dim, deadline, budget, rng)
                    f_refined = func(x_refined)
                    if f_refined < best_f:
                        best_f = f_refined
                        best_x = x_refined

        return best_x, best_f

    def _nelder_mead(self, func, x0, f0, dim, deadline, budget, rng):
        """Nelder-Mead simplex method with adaptive step sizes."""
        # Initialize simplex
        simplex = [x0.copy()]
        fvals = [f0]

        # Initial step size based on problem scale
        step = 0.5
        for i in range(dim):
            xi = x0.copy()
            xi[i] += step
            xi = np.clip(xi, -5, 5)
            fi = func(xi)
            simplex.append(xi)
            fvals.append(fi)

        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i].copy() for i in order]
        fvals = [fvals[i] for i in order]

        # Nelder-Mead iterations
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink
        max_iter = 500
        tol = 1e-12

        for it in range(max_iter):
            if time.time() > deadline:
                break

            # Check convergence
            if max(fvals) - min(fvals) < tol:
                break

            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)

            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
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
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5, 5)
                fc = func(xc)

                if fc < min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])

            # Re-sort
            order = np.argsort(fvals)
            simplex = [simplex[i].copy() for i in order]
            fvals = [fvals[i] for i in order]

        return simplex[0]
# EVOLVE-BLOCK-END
