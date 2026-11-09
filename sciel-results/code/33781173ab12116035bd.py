"""Optimizer for BBOB f14 (5-D) with a 500-call budget."""
import time
import numpy as np


class _CMAES:
    """Compact CMA-ES for box-constrained minimization."""

    def __init__(self, x0, f0, func, dim, lam, sigma0, deadline):
        self.func = func
        self.dim = dim
        self.lam = lam
        self.x = np.asarray(x0, dtype=float).copy()
        self.f = float(f0)
        self.best_x = self.x.copy()
        self.best_f = self.f
        self.sigma = float(sigma0)
        self.C = np.eye(dim)
        self.invC = np.eye(dim)
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.deadline = deadline
        self.chol = np.linalg.cholesky(self.C)
        self.chol_inv = np.linalg.cholesky(np.eye(dim))

        n = dim
        self.cc = (4 + n / 3.0) / (n + 4.0 + 2.0 * (n - 1) / (n + 1))
        self.cs = (n + 2 - self.cc) / (n + 6)
        self.mu = lam // 2
        w = np.log(self.mu + 1) - np.arange(self.mu)
        self.w = w / w.sum()
        self.mueff = 1.0 / self.w.sum()
        self.c1 = (2 + 2.0 / (n + 1.3)) / ((n + 2) ** 2 + self.mueff)
        self.cm = min(1.0 - self.c1,
                      (2 + 2.0 / (n + 1.3)) * (self.cs + 1) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1.0 + 2.0 * max(0.0, np.sqrt((self.mueff - 1) / (n + 1)) - 1.0) + self.cs
        self.chiN = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

    def _step(self):
        n = self.dim
        ys = self.sigma * self.chol @ np.random.randn(self.lam, n).T
        ys = np.clip(ys, -5.0, 5.0)
        ys -= ys.mean(axis=1, keepdims=True)
        ymean = ys.mean(axis=0)
        self.ps = (1.0 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.cc) * ymean
        self.pc = (1.0 - self.cc) * self.pc + (self.ps.T @ self.ps == self.ps @ self.ps) * \
                  (self.ps / self.sigma if self.sigma > 0 else self.ps)
        hsig = (np.linalg.norm(self.ps) / np.sqrt(1.0 - (1.0 - self.cs) ** 2)
                < (1.4 + self.cc / self.cs) * self.chiN)
        self.C = (1.0 + self.c1 * (2 - self.cm)) * self.C
        self.C += self.c1 * (self.pc @ self.pc - self.C)
        if hsig:
            self.C += self.cm * (ymean @ ymean - self.C)
        self.C = 0.5 * (self.C + self.C.T)
        # recompute cholesky safely
        try:
            self.chol = np.linalg.cholesky(self.C + 1e-9 * np.eye(n))
        except np.linalg.LinAlgError:
            self.C = 0.9 * self.C + 0.1 * np.eye(n)
            self.chol = np.linalg.cholesky(self.C + 1e-9 * np.eye(n))

    def run(self, budget):
        n = self.dim
        lam = self.lam
        mu = self.mu
        while budget > 0:
            if time.time() > self.deadline:
                return self.best_x, self.best_f
            xs = self.x + self.chol.T @ np.random.randn(lam, n)
            xs = np.clip(xs, -5.0, 5.0)
            fs = np.array([self.func(x) for x in xs], dtype=float)
            budget -= lam
            order = np.argsort(fs)
            selected = order[:mu]
            newx = (self.w[:, None] * xs[selected]).sum(axis=0)
            self.x = newx
            if fs[order[0]] < self.best_f:
                self.best_f = float(fs[order[0]])
                self.best_x = xs[order[0]].copy()
            self._step()
        return self.best_x, self.best_f


class Optimizer:
    """Multi-start CMA-ES with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin well under 330 s

        rng = np.random.default_rng(12345)
        starts = rng.uniform(-5.0, 5.0, size=(8, dim))

        # initial evaluations of the starts
        fvals = np.array([func(s) for s in starts], dtype=float)
        budget -= 8
        best_i = int(np.argmin(fvals))
        best_x = starts[best_i].copy()
        best_f = float(fvals[best_i])

        lam = 10
        sigma0 = 1.5

        # build a CMA-ES per start (lazy: only when it gets budget)
        algos = []
        for i in range(8):
            a = _CMAES(starts[i], float(fvals[i]), func, dim, lam, sigma0, deadline)
            a.best_x = starts[i].copy()
            a.best_f = float(fvals[i])
            algos.append(a)

        # allocation: proportional to (best_f - fvals[i]) with a floor
        floor = 0.05
        while budget > 0 and time.time() < deadline:
            # weights
            span = best_f - fvals
            span = np.clip(span, 0.0, None)
            weights = np.maximum(span, floor)
            weights = weights / weights.sum()
            # give at least one iteration to a random non-starved start
            for i in range(8):
                if budget < lam:
                    break
                if time.time() > deadline:
                    break
                if weights[i] < 1e-6:
                    continue
                # decide whether to spend here (stochastic by weight)
                if rng.random() < weights[i]:
                    bx, bf = algos[i].run(lam)
                    budget -= lam
                    if bf < best_f:
                        best_f = bf
                        best_x = bx.copy()
                    fvals[i] = bf
            # safety: if we keep looping with no progress, force progress
            if budget < lam:
                break

        return best_x, best_f
