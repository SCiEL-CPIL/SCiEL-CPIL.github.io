"""Black-box continuous optimizer for Rastrigin f3 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts and local polish for minimizing func on [-5,5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _local_polish(self, func, x0, budget_left, deadline, rng, best_fx):
        """Nelder-Mead style local refinement with budget limit."""
        dim = self.dim
        x = x0.copy()
        fx = best_fx
        calls = 0

        # Build initial simplex
        simplex = [x.copy()]
        for i in range(dim):
            xi = x.copy()
            step = 0.1 if abs(x[i]) < 0.5 else 0.05
            xi[i] += step
            xi = np.clip(xi, -5, 5)
            simplex.append(xi)

        fvals = np.empty(dim + 1)
        for i in range(dim + 1):
            if calls >= budget_left or time.time() > deadline:
                break
            fvals[i] = func(simplex[i])
            calls += 1
            if fvals[i] < fx:
                fx = fvals[i]
                x = simplex[i].copy()

        if calls >= budget_left or time.time() > deadline:
            return x, fx, calls

        alpha, gamma, rho = 1.0, 2.0, 0.5

        while calls < budget_left and time.time() <= deadline:
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i].copy() for i in order]
            fvals = fvals[order]

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            if calls >= budget_left or time.time() > deadline:
                break
            fr = func(xr)
            calls += 1
            if fr < fx:
                fx = fr
                x = xr.copy()

            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                if calls >= budget_left or time.time() > deadline:
                    break
                fe = func(xe)
                calls += 1
                if fe < fx:
                    fx = fe
                    x = xe.copy()
                simplex[-1] = xe if fe < fr else xr
                fvals[-1] = fe if fe < fr else fr
            elif fr >= fvals[-2]:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + 0.5 * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - 0.5 * (xr - centroid)
                xc = np.clip(xc, -5, 5)
                if calls >= budget_left or time.time() > deadline:
                    break
                fc = func(xc)
                calls += 1
                if fc < fx:
                    fx = fc
                    x = xc.copy()
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                simplex[-1] = xr
                fvals[-1] = fr

            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc

            # Shrink if stuck
            if np.max(fvals) - np.min(fvals) < 1e-10:
                for i in range(1, dim + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    if calls >= budget_left or time.time() > deadline:
                        break
                    fvals[i] = func(simplex[i])
                    calls += 1
                    if fvals[i] < fx:
                        fx = fvals[i]
                        x = simplex[i].copy()
                if np.max(fvals) - np.min(fvals) < 1e-12:
                    break

        return x, fx, calls

    def _cma_es_run(self, func, x0, sigma0, budget_left, deadline, rng, best_fx):
        """Run CMA-ES from x0 with initial sigma0, using budget_left evaluations."""
        dim = self.dim
        calls = 0

        # CMA-ES parameters
        lam = 12
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)
        cc = (4 + mu_eff / (dim + 4)) / (dim + 4 + 2 * mu_eff / (dim + 4))
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        c1 = (2 / ((dim + 1.3) ** 2 + mu_eff)) * (2 - cc + cs)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        x = x0.copy()
        sigma = sigma0
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        best_fx_local = best_fx
        best_x_local = x0.copy()

        while calls < budget_left and time.time() <= deadline:
            # Generate population
            B = rng.standard_normal((lam, dim))
            L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            pop = x[None, :] + sigma * (B @ L.T)
            pop = np.clip(pop, -5, 5)

            # Evaluate population
            fvals = np.empty(lam)
            for i in range(lam):
                if calls >= budget_left or time.time() > deadline:
                    break
                fvals[i] = func(pop[i])
                calls += 1
                if fvals[i] < best_fx_local:
                    best_fx_local = fvals[i]
                    best_x_local = pop[i].copy()

            if calls >= budget_left or time.time() > deadline:
                break

            # Sort by fitness
            order = np.argsort(fvals)
            pop_sorted = pop[order]
            fvals_sorted = fvals[order]

            # Update mean
            x_old = x.copy()
            x = sum(weights[i] * pop_sorted[i] for i in range(mu))
            x = np.clip(x, -5, 5)

            # Update evolution paths
            diff = (x - x_old) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (L.T @ diff)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (calls // lam + 1))
                    / np.sqrt(dim) < 1.4 + 2 / (dim + 1))

            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * diff

            # Update covariance
            delta = (1 + 2 * (hsig - 1) * c1 / cmu) if cmu > 0 else 1
            outer = np.zeros((dim, dim))
            for i in range(mu):
                outer += weights[i] * np.outer(pop_sorted[i] - x_old, pop_sorted[i] - x_old)
            C = (1 + c1 * delta - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu / sigma ** 2 * outer

            # Ensure positive definiteness
            C = (C + C.T) / 2
            try:
                np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            except np.linalg.LinAlgError:
                C = (C + 1e-5 * np.eye(dim)) / 2

            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

            # Stopping criterion
            if sigma < 1e-12 or (np.max(fvals_sorted) - np.min(fvals_sorted)) < 1e-12:
                break

        return best_x_local, best_fx_local, calls

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 320.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_fx = np.inf
        total_calls = 0

        # Multiple starting points with different sigmas
        starts = [
            (np.zeros(dim), 1.0),
            (np.array([2.0] * dim), 0.5),
            (np.array([-2.0] * dim), 0.5),
            (np.array([1.0, -1.0, 1.0, -1.0, 1.0]), 0.8),
            (np.array([-1.0, 1.0, -1.0, 1.0, -1.0]), 0.8),
        ]

        # Allocate budget: 60% for CMA-ES restarts, 40% for local polish
        cma_budget_per_restart = max(50, int(budget * 0.6 / len(starts)))
        polish_budget = max(50, int(budget * 0.4))

        for x0, sigma0 in starts:
            if total_calls >= budget or time.time() > deadline:
                break

            remaining = budget - total_calls
            alloc = min(cma_budget_per_restart, remaining)

            x_best, fx_best, calls_used = self._cma_es_run(
                func, x0, sigma0, alloc, deadline, rng, best_fx
            )
            total_calls += calls_used

            if fx_best < best_fx:
                best_fx = fx_best
                best_x = x_best.copy()

            # Local polish on the best found so far
            if total_calls < budget and time.time() <= deadline:
                polish_alloc = min(polish_budget, budget - total_calls)
                x_polished, fx_polished, polish_calls = self._local_polish(
                    func, best_x, polish_alloc, deadline, rng, best_fx
                )
                total_calls += polish_calls

                if fx_polished < best_fx:
                    best_fx = fx_polished
                    best_x = x_polished.copy()

        # Final polish if budget remains
        if total_calls < budget and time.time() <= deadline:
            final_alloc = budget - total_calls
            x_final, fx_final, final_calls = self._local_polish(
                func, best_x, final_alloc, deadline, rng, best_fx
            )
            total_calls += final_calls

            if fx_final < best_fx:
                best_fx = fx_final
                best_x = x_final.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
