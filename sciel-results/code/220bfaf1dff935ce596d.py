"""CMA-ES + (1+1)-ES polish for the BBOB-style Schwefel f20 task."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES (lambda=4, mu=2) with deadline guard and a final (1+1)-ES polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        d = self.dim
        BUDGET = self.budget
        LO, HI = -5.0, 5.0
        t0 = time.time()
        DEADLINE = 325.0  # safety margin under the 330 s wall-clock limit

        def _time_ok():
            return (time.time() - t0) < DEADLINE

        rng = np.random.default_rng(12345)

        # ---- CMA-ES state ----
        x = np.array([0.0] * d)
        sigma = 1.5
        C = np.eye(d)
        B = np.eye(d)
        D = np.ones(d)
        pc = np.zeros(d)

        mu = 2
        weights = np.ones(mu)
        weights[0] = 1.0 + 0.5 * np.log(1.0 + 0.5 * (d - 1.0))
        weights = weights / weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1.0 + 2 * max(0.0, (np.sqrt(mu_eff + 1) - 1) / (d + 1))
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1,
                   2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))
        chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        best_x = x.copy()
        best_f = float("inf")
        used = 0

        def _eval(pt):
            nonlocal best_x, best_f, used
            f = float(func(np.clip(pt, LO, HI)))
            used += 1
            if f < best_f:
                best_f = f
                best_x = np.array(pt, copy=True)
            return f

        _eval(x)
        max_iter = max(1, (BUDGET - 1) // mu)

        for it in range(max_iter):
            if not _time_ok():
                break
            # sample mu points
            z = rng.standard_normal((mu, d))
            y = np.array([x + sigma * (B @ (D * zi).T).T for zi in z])
            # evaluate
            fy = np.empty(mu)
            for i in range(mu):
                if not _time_ok() or used >= BUDGET:
                    break
                fy[i] = _eval(y[i])
            if used >= BUDGET:
                break
            # select best mu (all mu here)
            order = np.argsort(fy)
            ysel = y[order]
            wsum = np.sum(weights)
            yw = np.sum(weights[:, None] * ysel, axis=0)
            # covariance update
            v = (yw - x) / sigma
            pc = (1 - c_c) * pc + c_c * v
            hsig = (np.sqrt((1 - (1 - c_c) ** (2 * (it + 1))) * c_c * (2 - c_c) * mu_eff) / np.sqrt(d)) * np.linalg.norm(pc)
            delta_h = 1.4 + 2 / (d + 1)
            C = ((1 + c_1 * (delta_h - 1) - c_1 - c_mu) * C
                 + c_1 * np.outer(pc, pc)
                 + c_mu * np.sum(weights[:, None, None] * z[order] * np.transpose(z[order], (1, 0, 2)), axis=0))
            D2, B = np.linalg.eigh(C)
            D2 = np.sqrt(np.maximum(D2, 1e-15))
            D, B = D2, B
            # sigma adaptation
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(v) / chiN - 1.0))
            sigma = float(np.clip(sigma, 1e-12, 1e6))
            x = yw

        # ---- final (1+1)-ES polish on best point ----
        if _time_ok() and best_f != float("inf"):
            px = best_x.copy()
            pf = best_f
            psigma = max(1e-3, sigma * 0.05)
            while used < BUDGET and _time_ok():
                y = np.clip(px + psigma * rng.standard_normal(d), LO, HI)
                fy = _eval(y)
                if fy < pf:
                    px, pf = y, fy
                    psigma *= 1.3
                else:
                    psigma *= 0.8
                if psigma < 1e-10:
                    break
            best_x, best_f = px, pf

        return best_x, best_f
# EVOLVE-BLOCK-END
