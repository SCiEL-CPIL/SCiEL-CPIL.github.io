"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with Sobol global sampling for Schaffers F7."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _nelder_mead(self, func, x0, max_iter, deadline):
        """Nelder-Mead local optimizer with time guard."""
        dim = self.dim
        n = dim + 1
        # Initialize simplex
        simplex = np.zeros((n, dim))
        simplex[0] = x0
        for i in range(1, n):
            simplex[i] = x0.copy()
            simplex[i][i - 1] += 0.1

        fvals = np.array([func(simplex[i]) for i in range(n)])
        best_x = simplex[np.argmin(fvals)].copy()
        best_f = fvals.min()
        calls = n

        for _ in range(max_iter):
            if time.time() > deadline:
                break

            # Sort simplex by function value
            idx = np.argsort(fvals)
            simplex = simplex[idx]
            fvals = fvals[idx]

            # Centroid of all but worst
            centroid = simplex[:-1].mean(axis=0)

            # Reflection
            alpha = 1.0
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls += 1

            if fr < fvals[0]:
                # Expansion
                gamma = 2.0
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls += 1
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                    if fe < best_f:
                        best_f, best_x = fe, xe.copy()
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
                    if fr < best_f:
                        best_f, best_x = fr, xr.copy()
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
                if fr < best_f:
                    best_f, best_x = fr, xr.copy()
            else:
                # Contraction
                rho = 0.5
                if fr < fvals[-1]:
                    xc = centroid + rho * (simplex[-1] - centroid)
                else:
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls += 1
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                    if fc < best_f:
                        best_f, best_x = fc, xc.copy()
                else:
                    # Shrink
                    for i in range(1, n):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if fvals[i] < best_f:
                            best_f, best_x = fvals[i], simplex[i].copy()

            # Convergence check
            spread = np.max(fvals) - np.min(fvals)
            if spread < 1e-12:
                break

        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Sobol sequence for global sampling
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True)

        # Phase 1: Global exploration with Sobol points
        # Use ~100 points for global scan
        n_global = 100
        points = sampler.random(n_global)
        points = points * 10.0 - 5.0  # map to [-5, 5]^dim

        best_x = None
        best_f = np.inf
        calls = 0

        for p in points:
            if time.time() > deadline:
                break
            fp = func(p)
            calls += 1
            if fp < best_f:
                best_f = fp
                best_x = p.copy()

        # Phase 2: Local refinement from top candidates
        # Evaluate more points and keep track of best
        # Use remaining budget for local search from best point
        remaining_budget = budget - calls
        if remaining_budget > 0 and best_x is not None:
            # Try Nelder-Mead from the best global point
            nm_calls = min(remaining_budget, 200)
            best_x, best_f, nm_used = self._nelder_mead(func, best_x, nm_calls, deadline)
            calls += nm_used

            # If we have budget left, try a few more local searches from
            # other promising global points
            remaining = budget - calls
            if remaining > 20:
                # Sort global points by function value (we didn't store all, so re-evaluate top few)
                # Actually, let's just do a few more random local searches
                rng = np.random.default_rng(42)
                for _ in range(min(3, remaining // 30)):
                    if time.time() > deadline:
                        break
                    # Perturb the best point
                    x_try = np.clip(best_x + rng.normal(0, 0.5, dim), -5, 5)
                    ft = func(x_try)
                    calls += 1
                    if ft < best_f:
                        best_f = ft
                        best_x = x_try.copy()
                    # Quick local search
                    if remaining > 40:
                        bx, bf, used = self._nelder_mead(func, x_try, min(40, budget - calls), deadline)
                        calls += used
                        if bf < best_f:
                            best_f = bf
                            best_x = bx.copy()

        # Final polish: very tight local search
        remaining = budget - calls
        if remaining > 10 and best_x is not None:
            bx, bf, used = self._nelder_mead(func, best_x, min(remaining, 50), deadline)
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
