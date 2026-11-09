"""Optimizer for the BBOB-style f5 (Linear slope) black-box task."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class _CMAES:
    """A compact, fast CMA-ES implementation for box-bounded continuous problems."""

    def __init__(self, x0, sigma0, func, dim, bounds, rng, max_iter=None):
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.rng = rng
        self.max_iter = max_iter

        self.x = np.array(x0, dtype=float)
        self.f = self.func(self.x)
        self.sigma = float(sigma0)
        self.C = np.eye(dim)
        self.C_sqrt = np.eye(dim)
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)

        n = dim
        self.lam = 4 + int(3 * np.log(n))
        self.mu = self.lam // 2
        self.w = np.log(np.arange(1, self.mu + 1))
        self.w = self.w / self.w.sum()
        self.mueff = 1.0 / self.w.sum()
        self.c1 = (2 + (2.0 / (n + 1.3)) ** 2) / ((n + 2) ** 2 + 2)
        self.cm = min(1 - self.c1, (2 + (2.0 / (n + 1.3)) ** 2) / ((n + 2) ** 2 + 2))
        self.ccs = (4 + n / 4) / (n + 4 + 4 / n)
        self.csig = (1 + 2 / (n + 1) + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.csig
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        self._nfe = 1

    def _project(self, x):
        return np.clip(x, self.bounds[0], self.bounds[1])

    def _eigh(self, C):
        w, V = np.linalg.eigh(C)
        w = np.maximum(w, 1e-12)
        return np.sqrt(w), V

    def _update_sqrt(self):
        w, V = self._eigh(self.C)
        self.C_sqrt = V @ (w[:, None] * V.T)

    def run(self):
        it = 0
        while True:
            if self.max_iter is not None and it >= self.max_iter:
                break
            it += 1
            if self.sigma < 1e-12:
                break
            B, D = self._eigh(self.C)
            # generate offspring
            ys = []
            for _ in range(self.lam):
                z = self.rng.standard_normal(self.dim)
                ys.append(self._project(self.x + self.sigma * (B @ (D * z))))
            ys = np.array(ys)
            fs = np.array([self.func(y) for y in ys])
            self._nfe += self.lam
            order = np.argsort(fs)
            Y = ys[order]
            F = fs[order]
            ymean = np.sum(self.w[:, None] * Y[: self.mu], axis=0)
            d = ymean - self.x
            self.ps = (1 - self.csig) * self.ps + np.sqrt(
                self.csig * (2 - self.csig) * self.mueff
            ) * d / self.sigma
            hsig = (
                np.linalg.norm(self.ps)
                / np.sqrt(1 - (1 - self.csig) ** (2 * it))
                / self.chiN
                < 1.4 + 2 / (self.dim + 1)
            )
            self.pc = (1 - self.cc) * self.pc + float(hsig) * np.sqrt(
                self.cc * (2 - self.cc) * self.mueff
            ) * d / self.sigma
            rank1 = np.outer(self.ps, self.ps)
            rankmu = np.sum(
                self.w[:, None] * np.outer(Y[: self.mu] - self.x, Y[: self.mu] - self.x),
                axis=0,
            )
            self.C = (
                (1 + self.cmu * (self.c1 - 1) - self.c1) * self.C
                + self.c1 * rank1
                + self.cmu * rankmu
            )
            self.x = self._project(ymean)
            self.f = self.func(self.x)
            self._nfe += 1
            self.sigma *= np.exp(
                self.csig / self.damps
                * (np.linalg.norm(self.ps) / self.chiN - 1)
            )
            if it % 2 == 0:
                self._update_sqrt()
        return self.x, self.f


class Optimizer:
    """Restarting CMA-ES with a wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        bounds = (-5.0, 5.0)
        deadline = time.time() + 250.0  # safety margin under the 330s limit
        base_rng = np.random.default_rng(20240517)
        seeds = [101, 7, 23, 42, 1234, 99, 555, 31415, 271828, 65536, 137, 888]

        # first evaluation of each run counts toward budget; keep >=1 spare.
        max_runs = max(1, budget - 1)
        base_iters = max(8, min(40, budget // 4))

        best_x = None
        best_f = np.inf
        remaining = budget
        n_runs = 0

        for seed in seeds:
            if remaining <= 1:
                break
            if time.time() > deadline - 15.0:
                break
            n_runs += 1
            rng = np.random.default_rng(seed)
            x0 = rng.uniform(bounds[0], bounds[1], dim)
            # initial step size: a fraction of the box diagonal
            sigma0 = 0.6 * (bounds[1] - bounds[0])
            iters = base_iters
            if n_runs == 1:
                # give the first run a generous share of the budget
                iters = max(base_iters, (budget - 1) // 3)
            cma = _CMAES(x0, sigma0, func, dim, bounds, rng, max_iter=iters)
            x, f = cma.run()
            if f < best_f:
                best_f = f
                best_x = x
            remaining -= cma._nfe
            if best_f <= 1e-9:
                break  # essentially solved

        # ensure we return a valid point
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
