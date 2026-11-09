"""Optimizer for BBOB f17 (Schaffer's F7) using island-based CMA-ES with Nelder-Mead refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Island-based CMA-ES with Nelder-Mead local refinement for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 300s limit, leave 15s margin

        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_fx = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            fx = float(func(x))
            calls_made += 1
            return fx

        def time_ok():
            return time.time() < deadline and calls_made < budget

        # Phase 1: Initial sampling for early progress
        points = [np.zeros(d)]
        for i in range(d):
            for val in [-3.0, -1.5, 1.5, 3.0]:
                x = np.zeros(d)
                x[i] = val
                points.append(x.copy())
        for _ in range(8):
            points.append(rng.uniform(-4, 4, d))

        for x in points:
            if not time_ok():
                break
            x = np.clip(x, -5, 5)
            fx = call_func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_fx = call_func(best_x)

        # Compact CMA-ES for a single island
        def cma_es_island(x0, sigma0, max_calls, rng_local):
            nonlocal best_x, best_fx, calls_made

            n = d
            lam = max(4, int(4 + 3 * np.log(n)))
            mu = lam // 2

            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mueff = 1 / np.sum(w ** 2)
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = (2 / ((n + 1.3) ** 2 + mueff)) * (1 - cc - 2 * (min(1, cs - 1) * (n - 1) / n))
            cmu = (2 / ((n + 2) ** 2 + mueff)) * (mueff - 2 + 2 / (n - 2)) - c1
            cmu = min(cmu, 1 - c1)

            sigma = sigma0
            mean = x0.copy()
            C = np.eye(n)
            C_eig = np.linalg.eigh(C)
            C_eigvals = C_eig[0]
            C_eigvecs = C_eig[1]
            C_sqrt = C_eigvecs @ np.diag(np.sqrt(C_eigvals)) @ C_eigvecs.T

            ps = np.zeros(n)
            pc = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # Initial population
            pop = []
            fvals = []
            for i in range(lam):
                z = rng_local.standard_normal(n)
                x = mean + sigma * (C_sqrt @ z)
                x = np.clip(x, -5, 5)
                fx = call_func(x)
                pop.append(x)
                fvals.append(fx)

            idx = np.argmin(fvals)
            if fvals[idx] < best_fx:
                best_fx = fvals[idx]
                best_x = pop[idx].copy()

            order = np.argsort(fvals)
            pop = [pop[i] for i in order]
            fvals = [fvals[i] for i in order]

            max_iter = 50
            for iteration in range(max_iter):
                if not time_ok() or calls_made >= max_calls:
                    break

                old_mean = mean.copy()
                mean = sum(w[i] * pop[i] for i in range(mu))

                if np.linalg.norm(C_sqrt.T @ (mean - old_mean) / sigma) > (1.4 + 1 / cc) * cs * chi_n:
                    pc = (1 - cs) * pc
                else:
                    pc = (1 - cs) * pc + cs / chi_n * (C_sqrt.T @ (mean - old_mean)) / sigma

                hsig = (np.sqrt((1 - (1 - cc) ** (2 * (iteration + 1))) / (1 - (1 - cc) ** 2)) - 1 + cc) * chi_n
                if np.linalg.norm(pc) / chi_n < (1.4 + 1 / cc) * hsig:
                    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / chi_n * (mean - old_mean) / sigma
                else:
                    ps = (1 - cs) * ps

                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    y = (pop[i] - old_mean) / sigma
                    rank_mu += w[i] * np.outer(y, y)

                C = (1 - cs * c1 - cmu) * C + (1 - cs) * c1 * np.outer(pc, pc) + cmu * rank_mu
                C = (C + C.T) / 2
                try:
                    C_eig = np.linalg.eigh(C)
                    C_eigvals = np.maximum(C_eig[0], 1e-12)
                    C_eigvecs = C_eig[1]
                    C = C_eigvecs @ np.diag(C_eigvals) @ C_eigvecs.T
                    C_sqrt = C_eigvecs @ np.diag(np.sqrt(C_eigvals))
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    C_sqrt = np.eye(n)

                sigma *= np.exp(cs / (5 * n) * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-12)
                sigma = min(sigma, 10.0)

                if np.linalg.norm(mean - old_mean) < 1e-10 * max(1, sigma):
                    break

                new_pop = []
                new_fvals = []
                for i in range(lam):
                    z = rng_local.standard_normal(n)
                    x = mean + sigma * (C_sqrt @ z)
                    x = np.clip(x, -5, 5)
                    fx = call_func(x)
                    new_pop.append(x)
                    new_fvals.append(fx)

                idx = np.argmin(new_fvals)
                if new_fvals[idx] < best_fx:
                    best_fx = new_fvals[idx]
                    best_x = new_pop[idx].copy()

                order = np.argsort(new_fvals)
                pop = [new_pop[i] for i in order]
                fvals = [new_fvals[i] for i in order]

            return best_x, best_fx

        # Nelder-Mead local refinement
        def nelder_mead(x0, max_calls, rng_local):
            nonlocal best_x, best_fx, calls_made

            n = d
            alpha, gamma, rho = 1.0, 2.0, 0.5

            # Build initial simplex
            simplex = [x0.copy()]
            for i in range(n):
                xi = x0.copy()
                xi[i] += 0.5 if x0[i] < 4.5 else -0.5
                simplex.append(np.clip(xi, -5, 5))

            fvals = []
            for x in simplex:
                fx = call_func(x)
                fvals.append(fx)
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()

            max_iter = 30
            for _ in range(max_iter):
                if not time_ok() or calls_made >= max_calls:
                    break

                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Check convergence
                if fvals[0] - fvals[-1] < 1e-12:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fxr = call_func(xr)
                if fxr < best_fx:
                    best_fx = fxr
                    best_x = xr.copy()

                if fxr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fxe = call_func(xe)
                    if fxe < best_fx:
                        best_fx = fxe
                        best_x = xe.copy()
                    simplex[-1] = xe if fxe < fxr else xr
                    fvals[-1] = min(fxe, fxr)
                elif fxr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fxr
                else:
                    # Contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fxc = call_func(xc)
                    if fxc < best_fx:
                        best_fx = fxc
                        best_x = xc.copy()
                    if fxc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fxc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = call_func(simplex[i])
                            if fvals[i] < best_fx:
                                best_fx = fvals[i]
                                best_x = simplex[i].copy()

            return best_x, best_fx

        # Phase 2: Island-based multi-start CMA-ES
        # 3 islands, each with ~100 calls, then merge best
        n_islands = 3
        island_budget = min(100, (budget - calls_made) // n_islands)

        if island_budget < 20:
            # Fallback: single CMA-ES run
            if calls_made < budget and time_ok():
                cma_es_island(best_x.copy(), 1.0, budget, rng)
        else:
            island_results = []
            for island_i in range(n_islands):
                if not time_ok():
                    break

                # Each island starts from a different perturbation of the current best
                if island_i == 0:
                    x0 = best_x.copy()
                    sigma0 = 1.0
                else:
                    x0 = best_x + rng.normal(0, 1.0, d)
                    x0 = np.clip(x0, -5, 5)
                    sigma0 = 0.8

                rng_island = np.random.default_rng(42 + island_i * 1000)
                x_best_island, fx_best_island = cma_es_island(x0, sigma0, calls_made + island_budget, rng_island)
                island_results.append((fx_best_island, x_best_island.copy()))

            # Elite maintenance: keep the best island result
            if island_results:
                island_results.sort(key=lambda t: t[0])
                if island_results[0][0] < best_fx:
                    best_fx = island_results[0][0]
                    best_x = island_results[0][1].copy()

        # Phase 3: Nelder-Mead local refinement from the best point
        if calls_made < budget and time_ok():
            nm_budget = min(50, budget - calls_made)
            if nm_budget >= 10:
                rng_nm = np.random.default_rng(42 + 9999)
                nelder_mead(best_x.copy(), calls_made + nm_budget, rng_nm)

        # Phase 4: Final polish with small perturbations
        if calls_made < budget and time_ok():
            polish_calls = min(10, budget - calls_made)
            for _ in range(polish_calls):
                if not time_ok():
                    break
                y = best_x + 0.01 * rng.standard_normal(d)
                y = np.clip(y, -5, 5)
                fy = call_func(y)
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
