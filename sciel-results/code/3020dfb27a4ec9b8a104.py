"""Optimizer for BBOB f24 (Lunacek bi-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive step size and wall-clock guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit, 20s safety margin
        n = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)

        # Global best tracker
        best_x = np.full(n, np.nan)
        best_f = np.inf
        calls_made = 0

        def _time_ok():
            return (time.time() - start_time) < deadline

        def _eval(x):
            nonlocal calls_made
            calls_made += 1
            return func(x)

        # CMA-ES implementation
        def _cma_es(x0, sigma0, budget, rng):
            nonlocal best_x, best_f, calls_made
            n = self.dim
            x = np.array(x0, dtype=float)
            fx = _eval(x)
            if fx < best_f:
                best_f, best_x = fx, x.copy()
            sigma = sigma0

            # Population size
            lam = max(4, int(4 + 3 * np.log(n)))
            lam = min(lam, budget)

            # CMA-ES parameters
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / ((n + 1.3) + mueff)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mueff / n) / ((n + 4) + mueff)
            c_1 = 2 / ((n + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * ((mueff + 2) / (mueff - 2) - 1) * c_1)
            c_mu = max(c_mu, 0.0)

            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            it = 0
            stagnation = 0
            best_local_f = fx

            while calls_made < total_budget and it < budget and _time_ok():
                # Sample population
                B = rng.standard_normal((lam, n))
                # Transform using C^(1/2) via Cholesky
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback: use identity
                    L = np.eye(n)
                Y = (B @ L.T) * sigma
                pop = np.tile(x, (lam, 1)) + Y
                pop = np.clip(pop, -5, 5)

                # Evaluate population
                fvals = np.array([_eval(pop[i]) for i in range(lam)])
                if not _time_ok():
                    break

                # Update best
                idx_best = np.argmin(fvals)
                if fvals[idx_best] < best_f:
                    best_f, best_x = fvals[idx_best], pop[idx_best].copy()

                # Sort
                order = np.argsort(fvals)
                y_sorted = pop[order]
                f_sorted = fvals[order]

                # Update mean
                x_new = np.sum(weights[:, None] * y_sorted[:mu], axis=0)

                # Update ps
                inv_C_sqrt = np.linalg.inv(L)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * inv_C_sqrt @ (x_new - x)

                # Update pc
                h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (it + 1))) < (1 + c_sigma) * chi_n
                pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (x_new - x)

                # Update C
                delta = h_sigma * (1 - c_c) * c_c if h_sigma else 0.0
                C = (1 + c_1 + c_mu - c_c - delta) * C
                C += c_1 * (np.outer(pc, pc) - C)
                rank_mu = np.sum(weights[:mu, None] * np.outer(y_sorted[:mu] - x, y_sorted[:mu] - x), axis=0)
                C += c_mu * rank_mu

                # Ensure symmetry and positive definite
                C = 0.5 * (C + C.T)
                try:
                    np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = 0.9 * C + 0.1 * np.eye(n)
                    C = 0.5 * (C + C.T)

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-8)
                sigma = min(sigma, 20.0)

                # Move to new mean
                x = np.clip(x_new, -5, 5)
                fx = f_sorted[0]
                it += 1

                # Stagnation check
                if fx < best_local_f - 1e-10:
                    best_local_f = fx
                    stagnation = 0
                else:
                    stagnation += 1

                # Restart if stuck
                if stagnation > 50:
                    break

            return best_x, best_f

        # Multi-start strategy
        # Start 1: origin
        x0 = np.zeros(n)
        _cma_es(x0, 2.0, min(150, total_budget), rng)

        # Start 2: random point
        if _time_ok() and calls_made < total_budget:
            x0 = rng.uniform(-4, 4, n)
            _cma_es(x0, 1.5, min(150, total_budget - calls_made), rng)

        # Start 3: another random point if budget allows
        if _time_ok() and calls_made < total_budget:
            x0 = rng.uniform(-3, 3, n)
            _cma_es(x0, 1.0, min(150, total_budget - calls_made), rng)

        # Final refinement with (1+1)-ES if budget remains
        if _time_ok() and calls_made < total_budget and not np.isnan(best_x).any():
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            for _ in range(total_budget - calls_made):
                if not _time_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = _eval(y)
                if fy < fx:
                    x, fx = y, fy
                    if fy < best_f:
                        best_f, best_x = fy, x.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
