"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex optimizer with multi-start for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s

        dim = self.dim
        budget = self.budget

        # Nelder-Mead parameters
        alpha = 1.0    # reflection
        gamma = 2.0    # expansion
        rho = 0.5      # contraction
        sigma = 0.5    # shrink
        tol = 1e-10

        best_fx = np.inf
        best_x = np.zeros(dim)
        calls = 0

        def evaluate(x):
            nonlocal calls, best_fx, best_x
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        def nelder_mead(x0, max_calls):
            nonlocal calls, deadline
            if calls >= budget or time.time() > deadline:
                return

            # Initialize simplex
            simplex = np.zeros((dim + 1, dim))
            fvals = np.zeros(dim + 1)

            # First vertex is x0
            simplex[0] = x0
            fvals[0] = evaluate(simplex[0])

            # Other vertices: small perturbations
            for i in range(1, dim + 1):
                simplex[i] = x0.copy()
                simplex[i][i - 1] += 0.1  # Small step in each dimension
                fvals[i] = evaluate(simplex[i])

            # Main Nelder-Mead loop
            while calls < budget and time.time() <= deadline:
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # Check convergence
                if np.max(fvals) - np.min(fvals) < tol:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = evaluate(xr)

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = evaluate(xe)
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
                        fc = evaluate(xc)
                        if fc <= fr:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                fvals[i] = evaluate(simplex[i])
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                        fc = evaluate(xc)
                        if fc <= fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                fvals[i] = evaluate(simplex[i])

            # Return best point from this simplex
            order = np.argsort(fvals)
            return simplex[order[0]], fvals[order[0]]

        # Multi-start strategy
        # Start from center and several random points
        rng = np.random.RandomState(42)

        # First, start from center
        x0 = np.zeros(dim)
        nelder_mead(x0, budget)

        # If we still have budget, try a few more starts
        if calls < budget and time.time() < deadline:
            # Try a few random starts
            for _ in range(5):
                if calls >= budget or time.time() > deadline:
                    break
                x0 = rng.uniform(-4, 4, dim)
                nelder_mead(x0, budget)

        return best_x, best_fx
# EVOLVE-BLOCK-END
