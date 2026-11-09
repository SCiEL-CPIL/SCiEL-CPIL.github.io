"""Fresh design: parallel-restart CMA-ES for the Lunacek bi-Rastrigin (f24) task."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class _CMAES:
    """Covariance-Adaptive Evolution Strategy (standard, with step-length control).

    Parameters are the search-space parameters (center, sigma, covariance C).
    `func` is evaluated on the raw (unclipped) point; the outer Optimizer clips.
    """

    def __init__(self, x0, sigma0, dim, rng, func, deadline):
        self.dim = dim
        self.rng = rng
        self.func = func
        self.deadline = deadline

        self.x = np.asarray(x0, dtype=float).copy()
        self.sigma = float(sigma0)
        self.C = np.eye(dim)
        self.D = 1.0

        self.c1 = 1 / (4 * (dim + 1.7) ** 2 + 10)
        self.cmua = 2 / ((dim + 1.7) ** 2 + 13)
        self.c_sigma = (dim + 1.5) / (dim + 9)
        self.c_c = (4 + dim / dim) / (dim + 4)
        self.c_1 = 2 / ((dim + 1.3) ** 2 + 9)
        self.c_mu = min(1 - self.c_1,
                        2 * ((mu_eff + 0.1) / (dim + 2) - 1) /
                        ((dim + 1.3) ** 2 + 9))
        self.d_sigma = 1 + 2 * max(0, np.sqrt((dim - 1) / (dim + 3)) - 1) \
                       + self.c_sigma
        self.c_cs = (dim + 1.5 + self.c_sigma) / (dim + 9 + 2 * self.c_sigma)
        self.c_h = (1.7 + 2 / (dim + 1)) / (self.c_cs + 1)
        self.chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        n = dim + 1
        self.m = max(1, min(dim // 2, n - 1))
        self.mu = n // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= self.weights.sum()
        self.mu_eff = 1.0 / (self.weights ** 2).sum()

        self.p_sigma = np.zeros(dim)
        self.p_c = np.zeros(dim)
        self.hsig = 0.0
        self.ups = 0.0
        self.iter = 0
        self.budget = 0

    def _update(self, y, w, p_sigma, p_c, C, D, c_sigma, c_c, c_1, c_mu,
                d_sigma, c_cs, c_h, chiN, dim):
        self.iter += 1
        self.hsig = 1.0
        self.p_sigma = (1 - c_sigma) * self.p_sigma + \
            np.sqrt(c_sigma * (2 - c_sigma) * self.mu_eff) * (y / self.sigma)
        if np.linalg.norm(self.p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * self.iter)) \
                < (1.4 + 2 / (dim + 1)) * chiN:
            self.p_c = (1 - c_c) * self.p_c + \
                c_h * c_c * (1 - c_c) * (self.p_sigma / self.sigma)
            self.hsig = 1.0
        else:
            self.p_c = (1 - c_c) * self.p_c
            self.hsig = 0.0
        delta_h = (self.hsig - self.c_h) / (1 - self.c_c)
        self.C = (1 + self.c_1 * delta_h - self.c_1 - self.c_mu) * self.C \
            + self.c_1 * np.outer(self.p_c, self.p_c) \
            + self.c_mu * (w @ np.outer(y, y))
        D *= np.exp((self.c_1 / (2 * dim)) * (delta_h - 1)
                    + (self.c_mu / (2 * dim)) * (w @ (np.linalg.norm(y, axis=1) ** 2)
                                                 / self.sigma ** 2 - 2))
        self.sigma *= np.exp(self.c_sigma / self.d_sigma *
                             (np.linalg.norm(self.p_sigma) / chiN - 1))
        self.C = 0.5 * (self.C + self.C.T)

    def step(self):
        """Perform one generation; return (best_x, best_f, used_evals)."""
        dim = self.dim
        n = dim + 1
        z = self.rng.standard_normal((n, dim))
        y = (self.D * z) @ np.linalg.cholesky(self.C).T
        x = self.x + self.sigma * y
        fx = np.empty(n)
        for i in range(n):
            fx[i] = self.func(x[i])
            self.budget += 1
            if self.deadline is not None and time.perf_counter() >= self.deadline:
                return self.x, self._best_f, self.budget
        order = np.argsort(fx)
        y = y[order]
        fx = fx[order]
        w = self.weights
        self.x = self.x + self.sigma * (w @ y)
        self._best_f = fx[0]
        self._update(y[:self.mu], w, self.p_sigma, self.p_c, self.C, self.D,
                     self.c_sigma, self.c_c, self.c_1, self.c_mu,
                     self.d_sigma, self.c_cs, self.c_h, self.chiN, dim)
        return self.x, fx[0], n

    _best_f = np.inf


class Optimizer:
    """Parallel-restart CMA-ES minimizing func on [-5, 5]^dim."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start = time.perf_counter()
        # 330 s wall-clock kill; leave a generous safety margin.
        deadline = start + 285.0

        def clipped(x):
            return func(np.clip(x, -5.0, 5.0))

        rng = np.random.default_rng(20240517)
        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # --- Phase 1: cheap multi-start random search to locate basins. ---
        n_rand = min(10, budget)
        for _ in range(n_rand):
            if time.perf_counter() >= deadline:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            f = clipped(x)
            update_best(x, f)

        # --- Phase 2: parallel CMA-ES restarts from the top distinct seeds. ---
        # Build a diverse set of starting points from the random phase.
        cand = []
        for _ in range(0):
            pass
        # Re-collect candidate seeds by re-running a small deterministic grid
        # of the best random points found (kept implicitly via best_f only).
        # Instead, restart from a handful of well-spread points:
        seeds = []
        seeds.append(np.array([-4.0, -4.0, -4.0, -4.0, -4.0])[:dim])
        seeds.append(np.array([4.0, 4.0, 4.0, 4.0, 4.0])[:dim])
        seeds.append(np.zeros(dim))
        seeds.append(rng.uniform(-5.0, 5.0, dim))
        seeds.append(rng.uniform(-5.0, 5.0, dim))
        seeds.append(rng.uniform(-5.0, 5.0, dim))

        # Track how much budget each restart may consume.
        per_restart = max(20, budget // (len(seeds) + 2))
        for s in seeds:
            if time.perf_counter() >= deadline:
                break
            cma = _CMAES(s, 1.5, dim, rng, clipped, deadline)
            cma._best_f = clipped(s)
            update_best(s, cma._best_f)
            while True:
                x, f, _ = cma.step()
                update_best(x, f)
                if cma.budget >= per_restart:
                    break
                if time.perf_counter() >= deadline:
                    break

        # --- Phase 3: final local CMA-ES polish around the global best. ---
        if best_x is not None and time.perf_counter() < deadline:
            cma = _CMAES(best_x, 0.5, dim, rng, clipped, deadline)
            cma._best_f = best_f
            while time.perf_counter() < deadline:
                x, f, _ = cma.step()
                update_best(x, f)

        return best_x, best_f
# EVOLVE-BLOCK-END
