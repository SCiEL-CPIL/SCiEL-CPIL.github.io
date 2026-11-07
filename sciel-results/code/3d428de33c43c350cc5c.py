"""Optimizer for BBOB f20 (Schwefel) on [-5,5]^5 with a time-based deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Multi-start CMA-ES portfolio for the BBOB Schwefel (f20) benchmark.

    Strategy:
      * Begin with a small random Latin-hypercube-style spread of seeds to
        escape the many local minima of Schwefel.
      * Run a lightweight CMA-ES (a population-based derivative-free method)
        from the most promising seed, adapting its covariance and step size.
      * Track the global best across all evaluations so early progress is
        recorded, and stop cleanly with a safety margin before the wall-clock
        limit (330 s).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Safety margin: never run past this fraction of the 330 s limit.
        self.deadline = time.time() + 300.0

    def __call__(self, func):
        rng = np.random.default_rng(0)
        n = self.dim
        budget = self.budget

        # Global bookkeeping
        best_x = None
        best_f = np.inf
        calls = 0

        def eval_point(x):
            nonlocal calls, best_x, best_f
            fx = func(np.asarray(x, dtype=float))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = np.asarray(x, dtype=float).copy()
            return fx

        # --- Phase 1: cheap global spread to find a good basin -------------
        # Latin-hypercube sampling over [-5,5]^n gives broad coverage with
        # fewer points than a plain grid while keeping per-axis resolution.
        spread = 24
        if budget > spread:
            idx = np.arange(spread)
            lh = np.empty((spread, n))
            for d in range(n):
                perm = rng.permutation(spread)
                # stratified positions within each 1/24 slice
                lh[:, d] = -5.0 + (perm + rng.random(spread)) * (10.0 / spread)
            for i in range(spread):
                if calls >= budget:
                    break
                eval_point(lh[i])

        # --- Phase 2: CMA-ES from the best seed found ----------------------
        if calls < budget and best_x is not None:
            self._run_cmaes(func, eval_point, best_x, best_f, rng,
                            budget, calls)

        return best_x, best_f

    # ------------------------------------------------------------------ #
    def _run_cmaes(self, func, eval_point, x0, f0, rng, budget, start_calls):
        """Lightweight CMA-ES with time guard."""
        n = self.dim
        # population size
        lam = max(4, int(4 + 3 * np.log(n)))
        # initial sigma: adapt to basin scale
        sigma = 1.5
        x = np.asarray(x0, dtype=float).copy()
        C = np.eye(n)
        sigma_log = np.log(sigma)

        # adaptation parameters
        mu = lam // 2
        w = np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / (w * w).sum()
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1,
                   2 * (mu_eff - 2 + 1 / mu_eff) /
                   ((n + 2) ** 2 + mu_eff))
        d_sigma = (1 + c_sigma + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1)
                   + c_sigma * c_c)
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        while True:
            # time guard with safety margin
            if time.time() > self.deadline - 3.0:
                break
            if start_calls >= budget:
                break

            # sample population
            B = rng.standard_normal((lam, n))
            # whitening via Cholesky of C
            L = np.linalg.cholesky(C)
            y = B @ L.T
            pop = np.empty((lam, n))
            vals = np.empty(lam)
            for i in range(lam):
                if start_calls >= budget:
                    break
                if time.time() > self.deadline - 3.0:
                    break
                xi = np.clip(x + sigma * y[i], -5.0, 5.0)
                vals[i] = eval_point(xi)
                pop[i] = xi
                start_calls += 1

            if start_calls >= budget or time.time() > self.deadline - 3.0:
                break

            # sort by objective
            order = np.argsort(vals)
            y_sorted = y[order]
            # recenter
            m = np.sum(w[:, None] * pop[order], axis=0)
            d = m - x
            # update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * (L.T @ d)
            x = m
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * d
            # sigma adaptation
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma_log += c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1)
            sigma = np.exp(sigma_log)
            # keep sigma sane
            if sigma > 10.0:
                sigma = 10.0
                sigma_log = np.log(sigma)
            if sigma < 1e-4:
                sigma = 1e-4
                sigma_log = np.log(sigma)
            # covariance update
            delta_sigma = (1 / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1)
            h_sigma = (np.linalg.norm(p_sigma) /
                       np.sqrt(1 - (1 - c_sigma) ** (2 * (start_calls - lam))) /
                       chi_n) < (1.4 + 2 / (n + 1)) * (1 - c_sigma)
            C = ((1 + c_1 * (2 - c_1 - c_mu) - c_mu) * C
                 + c_1 * np.outer(p_c, p_c)
                 - (c_1 * (2 - c_1 - c_mu) - c_mu) * (np.eye(n) if False else np.outer(p_c, p_c) * 0)
                 + c_mu * np.sum(w[:, None] * np.outer(y_sorted[:mu], y_sorted[:mu])
                                 * (1 + (1 - h_sigma) * c_sigma * (2 - c_sigma)) * delta_sigma, axis=0)
                 - (c_mu * (1 - h_sigma) * c_sigma * (2 - c_sigma) * delta_sigma) * C)
            # symmetrize
            C = 0.5 * (C + C.T)
            # ensure PSD
            C = np.linalg.cholesky(C + 1e-12 * np.eye(n)).T @ np.linalg.cholesky(C + 1e-12 * np.eye(n))
# EVOLVE-BLOCK-END
