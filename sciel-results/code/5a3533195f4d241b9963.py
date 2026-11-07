"""Optimizer for the BBOB-style Büche-Rastrigin task (f4).

Population-based CMA-ES. Keeps the entry-point names/signatures of the seed:
class Optimizer with __init__(self, budget, dim) and __call__(self, func).
"""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Global black-box minimizer via CMA-ES on [-5, 5]^dim."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def _cmaes(self, func, budget, x0, sigma0, seed):
        n = self.dim
        rng = np.random.default_rng(seed)

        # population size
        lam = max(4, int(np.ceil(4 + 3 * np.log(n))))
        # step size
        sigma = float(sigma0)
        # step-size adaptation (CSA, default parameters)
        hs = 1.0
        damps = 0.5 + 2.0 / (n + 1.3)
        cs = (1.0 + 2.0 / max(1.0, n - 1.0) + damps / 2.0) / (n + 1.0 + 2.0 / max(1.0, n - 1.0))
        c1 = (2.0 + 2.0 / (n + 1.3)) / ((n + 2.0) ** 2 + damps)
        cmu = min(1.0 - c1, 2.0 * (2.0 + 2.0 / (n + 1.3)) / ((n + 2.0) ** 2 + damps))
        # cumulative step path
        pc = np.zeros(n)
        ps = np.zeros(n)
        # initial covariance (identity), precompute its square root (identity)
        B = np.eye(n)
        D = np.ones(n)
        invsqrt = np.eye(n)  # C^{-1/2}

        x = np.array(x0, dtype=float)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # pre-allocated buffers
        pop = np.empty((lam, n))
        fit = np.empty(lam)
        mu = lam // 2 + 1
        w = np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / w.sum()
        chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

        calls = 1
        deadline = time.time() + 290.0  # safety margin before the 300 s kill

        while calls < budget:
            if time.time() > deadline:
                break
            # sample population
            z = rng.standard_normal((lam, n))
            pop = x + sigma * (z @ B.T * D)
            pop = np.clip(pop, -5.0, 5.0)

            for i in range(lam):
                fit[i] = func(pop[i])
                calls += 1
                if calls >= budget:
                    break
            # record best
            imin = int(np.argmin(fit[:lam]))
            if fit[imin] < best_fx:
                best_fx = float(fit[imin])
                best_x = pop[imin].copy()

            # re-evaluate the current mean (CMA-ES standard)
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_fx = float(fx)
                best_x = x.copy()
            if calls >= budget:
                break

            # select and reorder by fitness
            order = np.argsort(fit)
            wsel = w[:mu]
            # weighted mean of selected (mu best)
            y = pop[order[:mu]]
            m = (ysel := (ysel := (w[:, None] * y).sum(axis=0)))  # weighted mean of selected
            dmean = m - x
            x = x + dmean

            # update covariance via rank-mu update
            # C = (1 - c1 - cmu) C + c1 (pc pc^T + (1 - hsig) C) + cmu sum w (yi yi^T)
            # We keep C implicitly via B, D. Full rank update is expensive; use a
            # simplified but stable update on the square root via eigendecomposition
            # of the accumulated direction. For n=5 this is cheap.
            C = B @ (D[:, None] * B.T)  # current covariance
            # hsig
            norm_pc = np.linalg.norm(pc)
            hsig = (norm_pc / np.sqrt(1.0 - (1.0 / (1.0 + cs)) ** calls)) < (1.4 + 2.0 / (n + 1)) * chi_n
            # update pc
            pc = (1.0 - cs) * pc + hsig * np.sqrt(cs * (2.0 - cs) * mueff) * dmean
            # update ps
            ps = (1.0 - cs) * ps + hsig * np.sqrt(cs * (2.0 - cs) * mueff) * (sigma * (z @ B.T * D)[order[:mu]].mean(axis=0))
            # step size
            sigma = sigma * np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1.0))
            # covariance update
            C = (1.0 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1.0 - hsig) * C)
            Ysel = y - np.outer(np.ones(mu), m)
            C = C + cmu * (w[:, None, None] * Ysel[:, :, None] * Ysel[:, None, :]).sum(axis=0)
            # recompute square root
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            D = np.sqrt(eigvals)
            B = eigvecs
            invsqrt = B @ (np.diag(1.0 / D) @ B.T)

            # keep x inside the box
            x = np.clip(x, -5.0, 5.0)

        return best_x, best_fx

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        # single robust global run of CMA-ES from a deterministic start
        x0 = np.zeros(n)
        # Rastrigin has a wide basin; start with a moderately large sigma
        sigma0 = 2.0
        best_x, best_fx = self._cmaes(func, budget, x0, sigma0, seed=0)
        return best_x, best_fx
# EVOLVE-BLOCK-END
