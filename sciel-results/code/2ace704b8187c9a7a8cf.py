"""Population-based DE (global) -> CMA-ES (local) for BBOB f16 Weierstrass."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """DE for global exploration followed by CMA-ES for local adaptation."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0          # safety margin under the 300 s limit
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(0)
        lo, hi = -5.0, 5.0
        clip = np.clip

        def remaining():
            return budget - used[0]

        def ok():
            return time.time() < deadline and used[0] < budget

        used = [0]

        def eval1(x):
            used[0] += 1
            return float(func(np.asarray(x, dtype=float)))

        # ---------------- Phase 1: DE (global exploration) ----------------
        N = max(2, 2 * dim)
        pop = rng.uniform(lo, hi, (N, dim))
        fvals = np.empty(N)
        for i in range(N):
            fvals[i] = eval1(pop[i])

        best_i = int(np.argmin(fvals))
        x_best = pop[best_i].copy()
        f_best = fvals[best_i]

        CR = 0.5
        F = 0.8
        while ok():
            order = np.argsort(fvals)
            best_i = int(order[0])
            x_best = pop[best_i].copy()
            f_best = fvals[best_i]
            for i in range(N):
                if not ok():
                    break
                i1 = rng.integers(0, N)
                i2 = rng.integers(0, N)
                while i1 == i2 or i1 == i or i2 == i:
                    i1 = rng.integers(0, N)
                    i2 = rng.integers(0, N)
                trial = pop[i1] + F * (pop[best_i] - pop[i2])
                trial = clip(trial, lo, hi)
                mask = rng.random(dim) < CR
                if not mask.any():
                    mask[rng.integers(0, dim)] = True
                cand = np.where(mask, trial, pop[i])
                fc = eval1(cand)
                if fc <= fvals[i]:
                    pop[i] = cand
                    fvals[i] = fc
                    if fc < f_best:
                        f_best = fc
                        x_best = cand.copy()
                        best_i = i
        if not ok():
            return x_best, f_best

        # ---------------- Phase 2: CMA-ES (local adaptation) ----------------
        n = max(4, dim + int(round(3 * np.log(dim))))
        x = x_best.copy()
        fx = f_best
        sigma = 0.5
        C = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        c_sigma = 1.0 / (n + 3)
        c_c = (4 + n / dim) / ((n + 4) * (dim + 4))
        d_sigma = 1 + 2 * max(0.0, (np.sqrt((n + 1) / (dim + 1)) - 1) / (c_sigma * (dim + 3)))
        c_1 = (2 + (n / dim) ** 2) / ((dim + 2) ** 2 + n)
        c_mu = min(0.5, (2 + (n / dim) ** 2) / ((dim + 2) ** 2 + 2 * n))
        mu = n // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        for _ in range(budget - used[0]):
            if not ok():
                break
            z = rng.standard_normal((n, dim))
            xk = x + sigma * z
            fk = np.empty(n)
            for k in range(n):
                if not ok():
                    break
                fk[k] = eval1(clip(xk[k], lo, hi))
            if used[0] >= budget:
                break
            order = np.argsort(fk)
            idx = order[:mu]
            y = z[idx]
            yw = (w[:, None] * y).sum(axis=0)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * yw / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2)) < (1.4 + 2 / (dim + 1)) * chi_n
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (yw / sigma - ps)
            C = ((1 + c_1 * (2 - c_1) - c_mu) * C
                 + c_1 * np.outer(ps, ps)
                 + c_mu * (w[:, None] * y @ (y.T * w[:, None])) / (sigma ** 2))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = min(max(sigma, 1e-10), 20.0)
            x = x + (w * y).sum(axis=0) * sigma
            fx = fk[idx[0]]
            x = clip(x, lo, hi)
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
        return x_best, f_best
# EVOLVE-BLOCK-END
