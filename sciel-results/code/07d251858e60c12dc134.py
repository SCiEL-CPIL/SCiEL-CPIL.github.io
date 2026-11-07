"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid optimizer: CMA-ES for global search + Nelder-Mead for local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit

        # Wrap func to count calls and clip
        call_count = [0]

        def wrapped(x):
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            call_count[0] += 1
            return float(func(x))

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        rng = np.random.default_rng(123)

        # Phase 1: CMA-ES for global search
        def cma_es(x0, sigma, max_generations):
            nonlocal best_x, best_f
            n = dim
            mu = max(1, n // 2)
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / ((n + 1.3) ** 2 + mueff)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            c_1 = 2 / ((n + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            mean = np.array(x0, dtype=np.float64)
            sigma = float(sigma)
            ps = np.zeros(n)
            C = np.eye(n)
            pop_size = 2 * mu
            no_improve = 0

            for gen in range(max_generations):
                if time.time() > deadline - 15.0:
                    break
                if call_count[0] >= budget - 5:
                    break

                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                    D_sqrt = np.sqrt(eigvals)
                    Y = rng.standard_normal((pop_size, n))
                    X = mean[None, :] + sigma * (Y * D_sqrt[None, :]) @ eigvecs.T
                except:
                    Y = rng.standard_normal((pop_size, n))
                    X = mean[None, :] + sigma * Y

                X = np.clip(X, -5.0, 5.0)
                F = np.array([wrapped(xi) for xi in X])
                idx = np.argsort(F)
                X_selected = X[idx[:mu]]
                F_selected = F[idx[:mu]]

                if F_selected[0] < best_f:
                    best_f = F_selected[0]
                    best_x = X_selected[0].copy()
                    no_improve = 0
                else:
                    no_improve += 1

                new_mean = np.sum(weights[:, None] * X_selected, axis=0)
                new_mean = np.clip(new_mean, -5.0, 5.0)

                diff_mean = (new_mean - mean) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * diff_mean

                delta_d = (1 + c_sigma * (2 - c_sigma)) * (np.linalg.norm(ps) ** 2 / n - 1)
                rank_one = np.outer(ps, ps)
                rank_mu = np.sum(weights[:, None, None] * X_selected[:, :, None] * X_selected[:, None, :], axis=0)
                C = (1 + c_1 * delta_d - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
                C = (C + C.T) / 2.0
                try:
                    eigvals_c, _ = np.linalg.eigh(C)
                    if np.any(eigvals_c < 1e-20):
                        C += 1e-10 * np.eye(n)
                except:
                    pass

                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                mean = new_mean

                if no_improve > 25:
                    break

            return best_x, best_f

        # Run CMA-ES with most of the budget
        cma_es(np.zeros(dim), 2.0, budget // 10)

        # Phase 2: Nelder-Mead for local polish
        if time.time() < deadline - 10.0 and call_count[0] < budget - 5:
            try:
                result = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'fatol': 1e-10, 'xatol': 1e-8}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5)
            except:
                pass

        # Phase 3: If budget remains, another CMA-ES with smaller sigma
        remaining = budget - call_count[0]
        if remaining > 2 * dim and time.time() < deadline - 10.0:
            cma_es(best_x.copy(), 0.5, remaining // 10)

        # Phase 4: Final Nelder-Mead polish
        if time.time() < deadline - 5.0 and call_count[0] < budget - 3:
            try:
                result = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'fatol': 1e-12, 'xatol': 1e-10}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5)
            except:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
