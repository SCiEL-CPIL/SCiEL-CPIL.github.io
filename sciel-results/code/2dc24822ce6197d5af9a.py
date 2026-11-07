"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Nelder-Mead simplex optimizer with adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Start from center (0, ..., 0) since the minimum is in [-4,4]^5
        # This is a reasonable prior given the problem structure
        x_best = np.zeros(dim)
        f_best = func(x_best)
        calls_made = 1

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink
        max_iter_per_restart = 100

        # Initial simplex: start at origin with small perturbations
        # Use a deterministic initial simplex to avoid randomness issues
        x0 = np.zeros(dim)
        f0 = f_best

        # Build initial simplex
        simplex = [x0.copy()]
        fvals = [f0]

        for i in range(dim):
            xi = x0.copy()
            xi[i] = 0.5  # small step in each dimension
            simplex.append(xi)
            fi = func(xi)
            fvals.append(fi)
            calls_made += 1

        # Track best
        idx_best = np.argmin(fvals)
        f_best = fvals[idx_best]
        x_best = simplex[idx_best].copy()

        # Main Nelder-Mead loop
        while calls_made < budget:
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Update best
            if fvals[0] < f_best:
                f_best = fvals[0]
                x_best = simplex[0].copy()

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls_made += 1

            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls_made += 1

                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                # Simple reflection
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                    fc = func(xc)
                    calls_made += 1
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                    fc = func(xc)
                    calls_made += 1

                if fc < min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fi = func(simplex[i])
                        calls_made += 1
                        fvals[i] = fi

        return x_best, f_best
# EVOLVE-BLOCK-END
