"""Rewritten optimizer: multi-start CMA-ES with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Derivative-free optimizer using multi-start CMA-ES."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0
        margin = 20.0  # safety margin before the 330s kill

        self._count = 0
        budget = self.budget
        dim = self.dim

        def cfunc(x):
            self._count += 1
            return func(x)

        def time_ok():
            return self._count < budget and (time.time() + margin) < deadline

        rng = np.random.default_rng(12345)

        # ---- Multi-start CMA-ES ----
        def cmaes(x0, sigma0, max_evals):
            d = dim
            x0 = np.asarray(x0, dtype=float).copy()
            fx = cfunc(x0)
            if self._count >= budget:
                return x0, fx

            # population size
            lam = max(4, int(4 + 3 * np.log(d)))
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mu_eff = 1.0 / (w @ w)
            c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
            c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))

            B = np.eye(d)
            D = np.ones(d)
            C = np.eye(d)
            pc = np.zeros(d)
            ps = np.zeros(d)
            sigma = sigma0
            chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

            while self._count < budget and (time.time() + margin) < deadline:
                # generate offspring
                ys = np.empty((lam, d))
                for i in range(lam):
                    z = rng.standard_normal(d)
                    ys[i] = x0 + sigma * (B @ (D * z))
                ys = np.clip(ys, -5.0, 5.0)
                fys = np.array([cfunc(y) for y in ys])

                # sort by fitness
                order = np.argsort(fys)
                ys = ys[order]
                fys = fys[order]

                # update best
                if fys[0] < fx:
                    x0, fx = ys[0], fys[0]

                # weighted mean of best mu
                m = (w[:mu] * ys[:mu]).T

                # step-size control
                x0_old = x0
                x0 = m
                if np.linalg.norm(m - x0_old) > 0:
                    ps = ((1 - c_sigma) * ps +
                          np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) /
                          np.linalg.norm(m - x0_old) * (B @ (D * (m - x0_old))))
                if np.linalg.norm(ps) / chi_d > np.sqrt(1 - (1 - c_sigma) ** (2 / max_evals)):
                    sigma /= np.exp(c_sigma * d_sigma * (np.linalg.norm(ps) / chi_d - 1))

                # covariance matrix adaptation
                hs = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) <
                      (1.4 + 2 / (d + 1)) * chi_d)
                pc = (1 - c_c) * pc + hs * c_c * (m - x0_old) / sigma
                v = (pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (m - x0_old) / sigma)
                C = ((1 + c_c * (2 - c_c) - c_1 - c_mu) * C
                     + c_c * np.outer(v, v)
                     + c_1 * (np.outer(pc, pc) - C)
                     + c_mu * sum(w[i] * np.outer(ys[i] - x0_old, ys[i] - x0_old) / sigma ** 2
                                 for i in range(mu)))

                # ensure symmetry and eigen decomposition
                C = 0.5 * (C + C.T)
                try:
                    D2, B2 = np.linalg.eigh(C)
                    D2 = np.sqrt(np.maximum(D2, 1e-20))
                except np.linalg.LinAlgError:
                    B2, D2 = np.eye(d), np.ones(d)
                    C = np.eye(d)
                B, D = B2, D2
                sigma = max(sigma, 1e-12)

            return x0, fx

        # ---- Phase 1: broad multi-start exploration ----
        n_starts = 8
        per_start = max(20, (budget - 1) // n_starts)
        best_x = None
        best_f = np.inf

        # deterministic spread of starting points + random
        starts = []
        for k in range(n_starts):
            if k == 0:
                s = np.zeros(dim)
            elif k == 1:
                s = rng.uniform(-4, 4, dim)
            else:
                s = rng.uniform(-4.5, 4.5, dim)
            starts.append(s)

        for s in starts:
            if self._count >= budget or (time.time() + margin) >= deadline:
                break
            x0, f0 = cmaes(s, 1.5, per_start)
            if f0 < best_f:
                best_f, best_x = f0, x0.copy()

        # ---- Phase 2: local refinement around the best point ----
        if best_x is not None and self._count < budget and (time.time() + margin) < deadline:
            x0, f0 = cmaes(best_x, 0.5, budget - self._count)
            if f0 < best_f:
                best_f, best_x = f0, x0.copy()

        # ---- Phase 3: final polish with tiny sigma if budget remains ----
        if best_x is not None and self._count < budget and (time.time() + margin) < deadline:
            x0, f0 = cmaes(best_x, 0.1, budget - self._count)
            if f0 < best_f:
                best_f, best_x = f0, x0.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
