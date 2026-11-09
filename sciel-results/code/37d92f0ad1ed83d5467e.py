"""Derivative-free population search (CMA-ES) with restarts and a deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class _CMA:
    """Compact CMA-ES for box-constrained black-box minimization."""

    def __init__(self, dim, x0, f0, sigma0, maxit, budget, func, deadline, rng):
        self.dim = dim
        self.budget = budget
        self.func = func
        self.deadline = deadline
        self.rng = rng
        self.x = x0.astype(float)
        self.f = f0
        self.sigma = sigma0
        self.maxit = maxit
        self.C = np.eye(dim)
        self.D = np.linalg.cholesky(self.C)
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.calls = 0

        n = dim
        self.lam = max(4, int(round((4 + int(np.log(n))) * 1.0)))
        self.mu = self.lam // 2
        w = np.log(self.mu + 1) - np.log(np.arange(1, self.lam + 1))
        w = w / w.sum()
        self.w = w
        self.mu_eff = 1.0 / np.sum(w[:self.mu] ** 2)
        self.cc = (4 + self.mu_eff / n) / (n + 4 + 2 * self.mu_eff / n)
        self.cs = (self.mu_eff + 2) / (n + self.mu_eff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mu_eff)
        self.cmua = min(1 - self.c1,
                        2 * (self.mu_eff - 2 + 1 / self.mu_eff) /
                        ((n + 2) ** 2 + self.mu_eff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (n + 1)) - 1) + self.cs
        self.chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

    def _sample(self):
        n = self.dim
        y = self.D @ self.rng.standard_normal(n)
        return np.clip(self.x + self.sigma * y, -5.0, 5.0)

    def _sqrt_C(self):
        C = self.C
        D = self.D
        n = self.dim
        for _ in range(3):
            try:
                D = np.linalg.cholesky(C)
                break
            except np.linalg.LinAlgError:
                C = 0.98 * C + 0.02 * np.eye(n)
        self.D = D

    def _update(self, xs, fs):
        n = self.dim
        order = np.argsort(fs)
        xs = xs[order]
        ws = self.w[:self.mu]
        m = np.sum(ws[:, None] * xs, axis=0)
        y = (m - self.x) / self.sigma
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * y
        hsig = (np.linalg.norm(self.ps) /
                np.sqrt(1 - (1 - self.cs) ** (2 * (self.maxit + 1)))) < \
               (1.4 + 2 / (n + 1)) * self.chi_n
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * y
        rank_mu = np.outer(y, y) * 0 + np.sum(ws[:, None] * (xs - self.x) / self.sigma, axis=0)
        rank_mu = np.outer(rank_mu, rank_mu)
        self.C = (1 + self.c1 * self.cs * (2 - self.cs) - self.c1 - self.cmua) * self.C \
                 + self.c1 * np.outer(self.pc, self.pc) + self.cmua * rank_mu
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chi_n - 1))
        self.sigma = float(np.clip(self.sigma, 1e-12, 1e6))
        self.x = m
        self._sqrt_C()

    def run(self):
        while self.calls < self.budget and self.maxit > 0:
            if time.time() > self.deadline:
                break
            self.maxit -= 1
            xs = np.empty((self.lam, self.dim))
            fs = np.empty(self.lam)
            for i in range(self.lam):
                if self.calls >= self.budget or time.time() > self.deadline:
                    break
                x = self._sample()
                self.calls += 1
                f = self.func(x)
                xs[i] = x
                fs[i] = f
            if np.any(np.isfinite(fs)):
                self._update(xs, fs)
        return self.x, self.f


class Optimizer:
    """Restarted CMA-ES on [-5, 5]^dim with a wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 295.0          # safety margin vs 300s per-eval / 330s kill
        budget = int(self.budget)
        dim = int(self.dim)
        rng = np.random.default_rng(1234)

        best_x = None
        best_f = np.inf
        used = 0
        t_start = time.time()
        while used < budget and (time.time() - t_start) < (deadline - t0):
            rem = budget - used
            # Adaptive restart schedule: shrink the sub-budget as time elapses.
            frac = (time.time() - t_start) / max(1e-9, (deadline - t0))
            sub = max(10, int(np.ceil(rem * (1.0 - 0.4 * frac))))
            sub = min(sub, rem)

            x0 = rng.uniform(-5, 5, dim)
            f0 = func(x0)
            used += 1
            if np.isfinite(f0) and f0 < best_f:
                best_f, best_x = f0, x0.copy()
            if used >= budget:
                break

            # CMA-ES sub-run; sigma0 scaled to the box size.
            cma = _CMA(dim, x0, f0, 1.0, sub - 1, sub, func, deadline, rng)
            cma.run()
            used += (cma.calls)
            if np.isfinite(cma.f) and cma.f < best_f:
                best_f, best_x = cma.f, cma.x.copy()

        if best_x is None:
            x = np.zeros(dim)
            try:
                func(x)
            except Exception:
                pass
            best_x = x
        return best_x, best_f
# EVOLVE-BLOCK-END
