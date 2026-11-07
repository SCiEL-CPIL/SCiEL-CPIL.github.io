"""Optimized BBOB Sphere f1 solver using CMA-ES + L-BFGS-B polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize as sp_minimize


class Optimizer:
    """CMA-ES with deadline guard and final L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 50s safety margin before 330s

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Start at center
        x_best = np.zeros(dim)
        fx_best = func(x_best)
        calls = 1

        # CMA-ES parameters
        lam = max(4 + int(3 * np.log(dim)), 4)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()

        c_sigma = (mu_eff + 2) / ((dim + 1) ** 2 + mu_eff)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        def mat_sqrt_eigh(A):
            w, V = np.linalg.eigh(A)
            w = np.maximum(w, 1e-12)
            return V @ np.diag(np.sqrt(w)) @ V.T

        gen = 0
        while calls < budget:
            if time.time() > deadline:
                break
            gen += 1

            B = mat_sqrt_eigh(C)
            pop = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = np.clip(x_best + sigma * (B @ z), -5, 5)
                fy = func(y)
                calls += 1
                pop.append((y, fy))
                if time.time() > deadline:
                    break

            if len(pop) < lam:
                last = pop[-1] if pop else (x_best, fx_best)
                pop.extend([last] * (lam - len(pop)))

            pop.sort(key=lambda p: p[1])
            if pop[0][1] < fx_best:
                x_best = pop[0][0].copy()
                fx_best = pop[0][1]

            mean_new = sum(weights[i] * pop[i][0] for i in range(mu))

            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (mean_new - x_best))
            h_sigma = 1 if np.linalg.norm(ps) / chi_n < (1.4 + 2 / (dim + 1)) * np.sqrt(1 - (1 - c_sigma) ** 2) else 0
            sigma *= np.exp(c_sigma / 5 * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-15, 1e3)

            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean_new - x_best) / sigma
            C = (1 + c_1 * h_sigma - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) - C + c_c * (2 - c_c) * np.outer(pc, pc))
            for i in range(mu):
                diff = pop[i][0] - x_best
                C += (c_mu * weights[i]) * np.outer(diff, diff)
            C = 0.5 * (C + C.T)
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-8 * np.eye(dim)

            x_best = mean_new

        # Final polish
        remaining = budget - calls
        if remaining > 5 and time.time() < deadline:
            try:
                bounds = [(-5, 5)] * dim
                res = sp_minimize(
                    func, x_best, method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': min(remaining // 2, 100), 'maxfun': remaining}
                )
                if res.fun < fx_best:
                    x_best = res.x.copy()
                    fx_best = res.fun
            except Exception:
                pass

        return x_best, fx_best
# EVOLVE-BLOCK-END
