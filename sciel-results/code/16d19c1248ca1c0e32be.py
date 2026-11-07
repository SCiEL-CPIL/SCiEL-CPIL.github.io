"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Stochastic multi-start Nelder-Mead optimizer for Schaffers F7."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        # We will do 10 independent runs, each with 50 function calls
        n_runs = 10
        calls_per_run = budget // n_runs  # 50

        # Deterministic starting points spread across [-4, 4]^5
        # Use a grid-like or Halton-like deterministic spread
        # For simplicity, use a deterministic set of 10 points
        # We'll use a simple deterministic spread: center + offsets
        rng = np.random.default_rng(42)

        # Generate 10 deterministic starting points
        # Use a quasi-random sequence (Halton-like) for good coverage
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        starts = []
        for i in range(1, n_runs + 1):
            point = np.array([halton(i, base) for base in range(2, dim + 2)])
            # Map from [0,1]^dim to [-4,4]^dim
            point = -4 + 8 * point
            starts.append(point)

        # Also add the center and a few random points for diversity
        starts = np.array(starts)

        best_x = None
        best_f = np.inf

        for start in starts:
            # Nelder-Mead simplex method
            # Simplex: dim+1 points
            # Initialize simplex around start
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = start
            for i in range(1, dim + 1):
                simplex[i] = start.copy()
                simplex[i, i - 1] += 0.5  # small perturbation

            # Evaluate initial simplex
            fvals = np.zeros(dim + 1)
            for i in range(dim + 1):
                fvals[i] = func(simplex[i])

            # Track calls used in this run
            calls_used = dim + 1
            max_calls = calls_per_run

            # Nelder-Mead iterations
            while calls_used < max_calls:
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + (centroid - simplex[-1])
                fxr = func(xr)
                calls_used += 1

                if fxr < fvals[0]:
                    # Expansion
                    xe = centroid + 2 * (centroid - simplex[-1])
                    fxe = func(xe)
                    calls_used += 1
                    if fxe < fxr:
                        simplex[-1] = xe
                        fvals[-1] = fxe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fxr
                elif fxr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fxr
                else:
                    # Contraction
                    if fxr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid + 0.5 * (simplex[-1] - centroid)
                    fxc = func(xc)
                    calls_used += 1
                    if fxc <= min(fxr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fxc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = func(simplex[i])
                            calls_used += 1

                # Update best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()

                # If we've exceeded budget, break
                if calls_used >= max_calls:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
