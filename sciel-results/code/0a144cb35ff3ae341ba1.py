"""Rewritten optimizer for BBOB f19 (Griewank-Rosenbrock) with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Population-based CMA-ES with restarts and a wall-clock guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin below 300s / 330s kill

        # ---------- helpers ----------
        def clip(x):
            return np.clip(x, -5.0, 5.0)

        # CMA-ES core (single run)
        def cmaes(x0, f0, sigma0, budget_cma, rng):
            n = dim
            # population size
            lam = max(4, int(4 + 3 * math.log(n)))
            # step size control
            cs = 1.0 / (n + 4)
            ds = 1.0 + 2 * max(0, (n / 2 - 1) / (n + 2))
            c1 = (2 + 2.0 / (n + 1.3)) / ((n + 2) ** 2 + 2)
            cmu = min(1 - c1,
                      2 * ((n / 8 + 1) - 2) / ((n + 2) ** 2 + 2))
            chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # state
            xmean = x0.copy()
            fmean = f0
            C = np.eye(n)
            D = np.linalg.cholesky(C)
            sigma = sigma0
            ps = np.zeros(n)
            phes = np.zeros(n)

            best_x = x0.copy()
            best_f = f0

            for it in range(budget_cma):
                if time.time() > deadline:
                    break
                # generate population
                z = rng.standard_normal((lam, n))
                y = (z @ D) / sigma
                x = xmean + sigma * y
                x = clip(x)

                # evaluate
                fx = np.array([func(xi) for xi in x])

                # update best
                if fx.min() < best_f:
                    best_f = fx.min()
                    best_x = x[fx.argmin()].copy()

                # sort
                order = np.argsort(fx)
                x = x[order]
                fx = fx[order]
                mu = lam // 2
                weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
                weights /= weights.sum()
                mueff = 1.0 / weights.sum()

                # update mean
                xmean_old = xmean
                xmean = xmean + sigma * np.sum(weights[:, None] * x[:mu], axis=0)
                xmean = clip(xmean)

                # update evolution paths
                pk = (xmean_old - xmean) / sigma
                ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * pk
                phes = (1 - ds) * phes + math.sqrt(ds * (2 - ds) * mueff) * (
                    np.linalg.norm(ps) / chi_n - 1)

                # update covariance
                delta = 1 + 2 * max(0, (phes - 1.1) / (n + 4)) + cs * (
                    (np.dot(ps, ps) / chi_n ** 2 - 1) * delta)
                rank_mu = (C / sigma ** 2) * 0.0
                for i in range(mu):
                    rank_mu += weights[i] * np.outer(
                        (x[i] - xmean_old) / sigma, (x[i] - xmean_old) / sigma)
                C = (1 + c1 * (delta - 1) + cmu) * C \
                    - (c1 + cmu) * C  # simplified; proper update below
                # Proper CMA update:
                C = (1 + c1 * (delta - 1) + cmu) * C \
                    - (c1 * (np.outer(ps, ps) / chi_n ** 2) + cmu * rank_mu / sigma ** 2)
                # Symmetrize
                C = 0.5 * (C + C.T)

                # sigma update
                sigma *= math.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(1e-10, min(sigma, 1e10))

                # Re-cholesky every few iterations
                if it % 2 == 0:
                    # Ensure PSD
                    eigvals, eigvecs = np.linalg.eigh(C)
                    if eigvals.min() < 1e-14:
                        C = eigvecs @ np.diag(np.maximum(eigvals, 1e-14)) @ eigvecs.T
                    D = np.linalg.cholesky(C)

            return best_x, best_f

        # ---------- main loop with restarts ----------
        rng = np.random.default_rng(0)
        best_x = None
        best_f = np.inf

        # Initial random restarts to find a good starting point
        n_initial = min(10, budget // 10)
        for i in range(n_initial):
            x0 = rng.uniform(-4, 4, dim)
            f0 = func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        # Run CMA-ES from the best point
        remaining = budget - n_initial
        if remaining > 5:
            x_best, f_best = cmaes(best_x.copy(), best_f, 1.0, remaining, rng)
            if f_best < best_f:
                best_f = f_best
                best_x = x_best

        # If we have time and budget left, try a few more restarts around best
        calls_made = n_initial + remaining
        if calls_made < budget and time.time() < deadline - 10:
            for restart in range(3):
                if time.time() > deadline:
                    break
                x0 = best_x + rng.standard_normal(dim) * 0.5
                x0 = clip(x0)
                f0 = func(x0)
                calls_made += 1
                if f0 < best_f:
                    best_f = f0
                    best_x = x0.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
