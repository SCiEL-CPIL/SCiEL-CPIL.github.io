"""Optimizer for BBOB-style f15 (Rastrigin rotated), 500-call budget, 5D."""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Population-based CMA-ES with restart + local polish + deadline guard."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)
        self.calls = 0
        self.deadline = time.time() + 320.0
        self.best_x = None
        self.best_f = np.inf

    def _eval(self, func, x):
        x = np.asarray(x, dtype=float).ravel()
        if x.shape[0] != self.dim:
            x = np.resize(x, self.dim)
        self.calls += 1
        f = float(func(x))
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f

    def __call__(self, func):
        d = self.dim
        rng = np.random.default_rng(12345)

        best_x, best_f = None, np.inf
        for _ in range(5):
            x0 = rng.uniform(-5.0, 5.0, d)
            f0 = self._eval(func, x0)
            if f0 < best_f:
                best_f, best_x = f0, x0.copy()

        def cmaes(x0, f0, budget_left, sigma0):
            n = budget_left
            lam = 12
            mu = 6
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1.0 / np.sum(w ** 2)
            c_sigma = (mueff + 2) / ((d + 2) * (mueff / 2 + 5))
            d_sigma = 1 + 2 * max(0, (mueff - 1) / (d + 1)) + c_sigma
            c_c = (4 + mueff / d) / ((d + 4) * (2 + mueff / d))
            c_1 = 2 / ((d + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
            chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

            x = np.asarray(x0, dtype=float).copy()
            sigma = sigma0
            B = np.eye(d); D = np.ones(d)
            pc = np.zeros(d); ps = np.zeros(d)
            C = np.eye(d)
            evals = 0
            stall = 0
            best_local = (f0, x0.copy())

            while evals < n and time.time() < self.deadline:
                Y = np.empty((lam, d)); F = np.empty(lam)
                for i in range(lam):
                    z = rng.standard_normal(d)
                    Y[i] = np.clip(x + sigma * (B @ (D * z)), -5.0, 5.0)
                    F[i] = self._eval(func, Y[i]); evals += 1
                order = np.argsort(F)
                Ys, Fs = Y[order], F[order]
                if Fs[0] < best_local[0]:
                    best_local = (float(Fs[0]), Ys[0].copy()); stall = 0
                else:
                    stall += 1

                y_w = np.zeros(d)
                for i in range(mu):
                    y_w += w[i] * (Ys[i] - x)
                x = np.clip(x + y_w, -5.0, 5.0)

                hsigma = np.sqrt((1 - np.exp(-2 / (n + 1))) + 1)
                ps = (1 - c_sigma) * ps + c_sigma * hsigma / max(sigma, 1e-12) * (B @ (D * y_w))
                if np.linalg.norm(ps) / chi_d < (1.4 + 2 / (d + 1)) * c_sigma:
                    sigma *= np.exp(c_sigma * d_sigma * (np.linalg.norm(ps) / chi_d - 1))

                pc = (1 - c_c) * pc + c_c * (y_w / max(sigma, 1e-12))
                outer = np.outer(y_w / max(sigma, 1e-12), y_w / max(sigma, 1e-12))
                C = (1 + c_1 + c_mu) * C + c_1 * (np.outer(pc, pc) - C) + c_mu * (outer - C)
                C = 0.5 * (C + C.T)
                if evals % lam == 0:
                    try:
                        eigvals, B = np.linalg.eigh(C)
                        eigvals = np.clip(eigvals, 1e-12, None)
                        D = np.sqrt(eigvals)
                    except np.linalg.LinAlgError:
                        B, D, C = np.eye(d), np.ones(d), np.eye(d)

                if sigma < 1e-6 or stall > 3 * lam:
                    x = np.clip(best_local[1] + 1.5 * rng.standard_normal(d), -5.0, 5.0)
                    sigma = max(0.5, sigma0 * 0.5)
                    B, D, C = np.eye(d), np.ones(d), np.eye(d)
                    pc, ps = np.zeros(d), np.zeros(d)
                    stall = 0
                    break

            return best_local

        result = cmaes(best_x, best_f, self.budget - self.calls, sigma0=2.5)
        if result[0] < self.best_f:
            self.best_f, self.best_x = result[0], result[1]

        remaining = self.budget - self.calls
        if remaining > 20 and time.time() < self.deadline - 2.0:
            def wrap(x):
                if self.calls >= self.budget or time.time() >= self.deadline:
                    return 1e300
                return self._eval(func, x)
            try:
                res = minimize(wrap, self.best_x.copy(), method="Nelder-Mead",
                               options={"maxiter": max(1, min(remaining // 3, 60)),
                                        "maxfev": remaining,
                                        "xatol": 1e-8, "fatol": 1e-12})
                if res.fun < self.best_f:
                    self.best_f, self.best_x = float(res.fun), np.asarray(res.x)
            except Exception:
                pass

        return self.best_x, self.best_f
