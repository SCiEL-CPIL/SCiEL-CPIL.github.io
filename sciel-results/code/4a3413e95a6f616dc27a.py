"""Rewritten optimizer: CMA-ES with adaptive restarts and L-BFGS-B polish for Lunacek f24."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES based optimizer with LHS initialization, adaptive restarts, and L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._calls = 0
        self._deadline = None

    def __call__(self, func):
        self._calls = 0
        self._deadline = time.time() + 280.0  # safety margin under 300s

        dim = self.dim
        budget = self.budget
        calls = [0]

        def f(x):
            if calls[0] >= budget:
                return 1e12
            if time.time() > self._deadline:
                return 1e12
            calls[0] += 1
            val = func(np.asarray(x, dtype=float))
            return float(val)

        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf

        # Generate Latin Hypercube samples for diverse initialization
        n_lhs = 20
        lhs_points = []
        for i in range(n_lhs):
            point = np.zeros(dim)
            for d in range(dim):
                perm = rng.permutation(n_lhs)
                interval = perm[i]
                point[d] = -5.0 + (interval + rng.random()) * (10.0 / n_lhs)
            lhs_points.append(point)

        # Add strategic points
        strategic_points = [
            np.zeros(dim),
            np.full(dim, 3.0),
            np.full(dim, -3.0),
            np.full(dim, 1.0),
            np.full(dim, -1.0),
            np.full(dim, 2.0),
            np.full(dim, -2.0),
            np.full(dim, 4.0),
            np.full(dim, -4.0),
            np.full(dim, 0.5),
            np.full(dim, -0.5),
        ]

        all_starts = lhs_points + strategic_points

        # Phase 1: Evaluate all starting points
        for x0 in all_starts:
            if calls[0] >= budget or time.time() > self._deadline:
                break
            fx0 = f(x0)
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

        # Phase 2: Proper CMA-ES implementation
        def cma_es(x0, sigma0, max_evals, seed=42):
            """Proper CMA-ES with population storage and sorting."""
            rng_local = np.random.default_rng(seed)
            n = len(x0)
            pop_size = max(4, int(4 + 3 * np.log(n)))
            lambda_ = pop_size
            mu = max(1, int(lambda_ / 2))
            weights = np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            cc = (4 + mu_eff / (n + 4)) / ((n + 2) + mu_eff / 2)
            cs = (2 + mu_eff / (n + 2)) / (n + 4 + mu_eff / 2)
            c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * (2 - mu_eff / (n + 2)) / ((n + 2) ** 2 + mu_eff))
            damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1) - 1)) + cs
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            sigma = sigma0
            mean = x0.copy()
            C = np.eye(n)
            sqrtC = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            best_local = x0.copy()
            best_local_f = f(mean)
            eval_count = 0
            stall_count = 0

            # Precompute B and D for sqrt(C)
            def update_sqrtC(C_mat):
                eigvals, eigvecs = np.linalg.eigh(C_mat)
                eigvals = np.clip(eigvals, 1e-20, None)
                return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

            sqrtC = update_sqrtC(C)

            for gen in range(max_evals // pop_size + 20):
                if eval_count >= max_evals or time.time() > self._deadline or calls[0] >= budget:
                    break

                # Generate population
                population = []
                for i in range(pop_size):
                    if eval_count >= max_evals or time.time() > self._deadline or calls[0] >= budget:
                        break
                    z = rng_local.standard_normal(n)
                    x = mean + sigma * (sqrtC @ z)
                    fx = f(x)
                    eval_count += 1
                    population.append((fx, x))
                    if fx < best_local_f:
                        best_local_f = fx
                        best_local = x.copy()

                if len(population) < 2:
                    break

                # Sort by fitness
                population.sort(key=lambda t: t[0])
                selected = [p[1] for p in population[:mu]]

                # Weighted mean
                new_mean = np.zeros(n)
                for i in range(mu):
                    new_mean += weights[i] * selected[i]

                # Update step size control
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (sqrtC.T @ (new_mean - mean))
                norm_ps = np.linalg.norm(ps)
                hsig = (norm_ps / (1 - (1 - cs) ** (gen + 1))) / chi_n < 1.4 + 2 / (n + 1)

                # Update evolution path
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (new_mean - mean) / sigma

                # Update covariance
                delta_hsig = (1 - hsig) * cc * (2 - cc)
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu * sum(weights[i] * np.outer(selected[i] - mean, selected[i] - mean) for i in range(mu))

                # Update mean
                mean = new_mean

                # Update sigma
                sigma *= np.exp(cs / damps * (norm_ps / chi_n - 1))
                sigma = max(1e-10, min(sigma, 10.0))

                # Update sqrt(C)
                sqrtC = update_sqrtC(C)

                # Stagnation check
                if np.linalg.norm(best_local - mean) < 1e-8 * sigma:
                    stall_count += 1
                else:
                    stall_count = 0

                if stall_count > 5:
                    break

            return best_local, best_local_f

        # Run CMA-ES from best point
        if best_x is not None and calls[0] < budget and time.time() < self._deadline:
            remaining = budget - calls[0]
            cma_budget = min(300, remaining)
            try:
                cma_x, cma_f = cma_es(best_x.copy(), 1.5, cma_budget, seed=123)
                if cma_f < best_f:
                    best_f = cma_f
                    best_x = cma_x.copy()
            except Exception:
                pass

        # Phase 3: Nelder-Mead local polish
        if calls[0] < budget and time.time() < self._deadline:
            try:
                res = minimize(
                    f,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'maxfev': 80, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 4: L-BFGS-B final polish
        if calls[0] < budget and time.time() < self._deadline:
            try:
                res = minimize(
                    f,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'maxfun': 60, 'ftol': 1e-10}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 5: If budget remains, try additional restarts with local search
        remaining_budget = budget - calls[0]
        if remaining_budget > 40 and time.time() < self._deadline:
            n_extra = min(3, remaining_budget // 40)
            for i in range(n_extra):
                if calls[0] >= budget or time.time() > self._deadline:
                    break
                # Try a few random restarts
                for _ in range(3):
                    if calls[0] >= budget or time.time() > self._deadline:
                        break
                    x_rand = rng.uniform(-4, 4, dim)
                    fx_rand = f(x_rand)
                    if fx_rand < best_f:
                        best_f = fx_rand
                        best_x = x_rand.copy()
                    # Quick Nelder-Mead from this point
                    if calls[0] < budget and time.time() < self._deadline:
                        try:
                            res = minimize(
                                f,
                                x_rand,
                                method='Nelder-Mead',
                                options={'maxiter': 50, 'maxfev': 15, 'xatol': 1e-4, 'fatol': 1e-4}
                            )
                            if res.fun < best_f:
                                best_f = res.fun
                                best_x = res.x.copy()
                        except Exception:
                            pass

        return best_x, best_f
# EVOLVE-BLOCK-END
