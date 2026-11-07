"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, fx0, max_iter, deadline):
        """Perform Nelder-Mead local optimization from x0."""
        dim = self.dim
        n = dim + 1  # number of vertices in simplex
        # Build initial simplex
        simplex = np.zeros((n, dim))
        simplex[0] = x0
        for i in range(1, n):
            simplex[i] = x0.copy()
            simplex[i][i - 1] = 0.25  # small perturbation along axis i-1
        
        fvals = np.full(n, np.inf)
        fvals[0] = fx0
        calls_used = 1
        for i in range(1, n):
            fvals[i] = func(simplex[i])
            calls_used += 1
            if time.time() > deadline:
                break

        # Sort simplex by function value
        def sort_simplex():
            nonlocal simplex, fvals
            idx = np.argsort(fvals)
            simplex = simplex[idx]
            fvals = fvals[idx]

        sort_simplex()
        best_x = simplex[0].copy()
        best_f = fvals[0]

        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink

        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if calls_used >= self.budget:
                break

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls_used += 1

            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls_used += 1
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (simplex[-1] - centroid)
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls_used += 1
                if fc <= min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls_used += 1
                        if time.time() > deadline:
                            break

            sort_simplex()
            if simplex[0][0] < best_x[0]:
                best_x = simplex[0].copy()
                best_f = fvals[0]

            # Check convergence
            if np.max(fvals) - np.min(fvals) < 1e-10:
                break

        return best_x, best_f, calls_used

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Global best
        best_x = None
        best_f = np.inf
        total_calls = 0

        # Multi-start strategy: try several starting points
        # Start with a few random points, then use best as seed for further refinement
        n_starts = min(10, max(3, budget // 100))
        max_iter_per_start = max(50, budget // n_starts - 5)

        # Initial random starts
        for i in range(n_starts):
            if time.time() > deadline:
                break
            if total_calls >= budget:
                break

            x0 = rng.uniform(-5, 5, dim)
            fx0 = func(x0)
            total_calls += 1

            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

            # Do Nelder-Mead from this point
            if time.time() > deadline:
                break
            if total_calls >= budget - 5:
                break

            x_opt, f_opt, calls_used = self._nelder_mead(
                func, x0, fx0, max_iter_per_start, deadline
            )
            total_calls += calls_used

            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

        # If we have budget left, do additional refinement from best point
        if best_x is not None and total_calls < budget and time.time() < deadline:
            remaining = budget - total_calls
            if remaining > 10:
                x_opt, f_opt, calls_used = self._nelder_mead(
                    func, best_x, best_f, remaining - 5, deadline
                )
                total_calls += calls_used
                if f_opt < best_f:
                    best_f = f_opt
                    best_x = x_opt.copy()

        # Final safety: if we haven't made any calls (shouldn't happen), make one
        if total_calls == 0:
            x = np.zeros(dim)
            fx = func(x)
            total_calls = 1
            best_x = x
            best_f = fx

        return best_x, best_f
# EVOLVE-BLOCK-END
