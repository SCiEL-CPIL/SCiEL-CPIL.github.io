"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with adaptive restarts for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, sigma, remaining_budget, time_limit):
        """
        Nelder-Mead simplex method.
        Returns (best_x, best_f, calls_used).
        """
        dim = self.dim
        # Initialize simplex
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = x0
        for i in range(dim):
            simplex[i + 1] = x0.copy()
            simplex[i + 1][i] += sigma

        fvals = np.empty(dim + 1)
        calls = 0

        # Evaluate initial simplex
        for i in range(dim + 1):
            fvals[i] = func(simplex[i])
            calls += 1

        best_idx = np.argmin(fvals)
        best_x = simplex[best_idx].copy()
        best_f = fvals[best_idx]

        # NM parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma_shrink = 0.5  # shrink
        min_sigma = 1e-8
        max_iter = remaining_budget - (dim + 1)

        for _ in range(max_iter):
            if time.time() > time_limit:
                break
            if calls >= remaining_budget:
                break

            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

            best_x = simplex[0].copy()
            best_f = fvals[0]

            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            worst = simplex[-1]
            xr = centroid + alpha * (centroid - worst)
            fr = func(xr)
            calls += 1

            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - worst)
                fe = func(xe)
                calls += 1
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
                    xc = centroid + 0.5 * (xr - centroid)
                    fc = func(xc)
                    calls += 1
                    if fc <= fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            fvals[i] = func(simplex[i])
                            calls += 1
                else:
                    # Inside contraction
                    xc = centroid - 0.5 * (worst - centroid)
                    fc = func(xc)
                    calls += 1
                    if fc <= fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            fvals[i] = func(simplex[i])
                            calls += 1

        return best_x, best_f, calls

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        time_limit = start_time + 290.0  # safety margin before 300s

        rng = np.random.default_rng(42)

        # Multi-start strategy: multiple initial points with varying sigma
        # Use a mix of deterministic and random starts
        best_x = None
        best_f = np.inf
        total_calls = 0

        # Define restart points
        # Start with center, then random points, then points near previous best
        n_restarts = 12
        remaining = budget

        for restart in range(n_restarts):
            if total_calls >= budget:
                break
            if time.time() > time_limit:
                break

            if restart == 0:
                # Start at origin
                x0 = np.zeros(dim)
                sigma = 2.0
            elif restart == 1:
                # Start at a random point
                x0 = rng.uniform(-4, 4, dim)
                sigma = 3.0
            elif restart == 2:
                # Start at another random point
                x0 = rng.uniform(-4, 4, dim)
                sigma = 2.5
            elif restart == 3:
                # Start near previous best with small perturbation
                if best_x is not None:
                    x0 = best_x + 0.5 * rng.standard_normal(dim)
                else:
                    x0 = rng.uniform(-3, 3, dim)
                sigma = 1.0
            else:
                # Subsequent restarts: mix of random and near-best
                if best_x is not None and rng.random() < 0.6:
                    # Perturb best
                    scale = 1.0 / (restart + 1)
                    x0 = best_x + scale * rng.standard_normal(dim)
                    sigma = max(0.1, scale * 0.5)
                else:
                    # New random start
                    x0 = rng.uniform(-4, 4, dim)
                    sigma = max(0.5, 3.0 / (restart + 1))

            # Clip to box
            x0 = np.clip(x0, -5, 5)
            sigma = min(sigma, 4.0)

            # Allocate budget for this restart
            # Give more budget to early restarts, less to later ones
            if restart < 4:
                alloc = max(30, (budget - total_calls) // 4)
            else:
                alloc = max(15, (budget - total_calls) // max(1, n_restarts - restart))
            alloc = min(alloc, remaining)

            # Run Nelder-Mead
            x_best, f_best, calls_used = self._nelder_mead(
                func, x0, sigma, alloc, time_limit
            )

            total_calls += calls_used
            remaining = budget - total_calls

            if f_best < best_f:
                best_f = f_best
                best_x = x_best

            # If we've converged well, we can stop early
            if best_f < 1e-10:
                break

        # Final refinement with a tight NM around the best point if budget remains
        if best_x is not None and remaining > 10 and time.time() < time_limit:
            x_ref, f_ref, calls_ref = self._nelder_mead(
                func, best_x, 0.1, remaining, time_limit
            )
            total_calls += calls_ref
            if f_ref < best_f:
                best_f = f_ref
                best_x = x_ref

        return best_x, best_f
# EVOLVE-BLOCK-END
