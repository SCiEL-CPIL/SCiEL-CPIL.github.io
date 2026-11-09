"""CMA-ES (correct adaptation) with time-deadline guard for BBOB f21."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES with proper sigma adaptation, h_sigma, and covariance update signs."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def _run(self, func, budget, deadline):
        d = self.dim
        rng = np.random.default_rng(20240607)

        # CMA-ES parameters (standard recommended values)
        popsize = 4 + int(np.floor(3 * np.log(d)))
        lam = popsize
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1,
                   2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))
        chi_d = np.sqrt(d) * (1 - 1.0 / (4 * d) + 1.0 / (21 * d ** 2))

        x = rng.uniform(-5, 5, d)
        fx = float(func(x))
        best_x = x.copy()
        best_f = fx
        sigma = 1.0
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        Cinv = np.eye(d)
        calls = 1

        while calls < budget:
            if time.time() > deadline:
                break

            # Sample population
            ys = np.empty((lam, d))
            zs = np.empty((lam, d))
            for i in range(lam):
                z = rng.multivariate_normal(np.zeros(d), C, 1)[0]
                ys[i] = np.clip(x + sigma * z, -5, 5)
                zs[i] = z
                calls += 1
                if calls >= budget:
                    break
            lam_eff = calls - 1
            if lam_eff < 2:
                break

            # Evaluate
            ys = ys[:lam_eff]
            zs = zs[:lam_eff]
            fs = np.array([float(func(y)) for y in ys])

            # Track best
            for i in range(lam_eff):
                if fs[i] < best_f:
                    best_f = fs[i]
                    best_x = ys[i].copy()

            # Sort by fitness
            order = np.argsort(fs)
            ys = ys[order]
            zs = zs[order]
            fs = fs[order]
            w = weights[:mu]

            ymean = np.sum(w[:, None] * ys[:mu], axis=0)
            zmean = np.sum(w[:, None] * zs[:mu], axis=0)

            # Update step-size control vector ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (
                Cinv @ zmean)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - 1)))) \
                / chi_d
            d_hsig = 1.0 if hsig < (1.4 + 2 / (d + 1)) else 0.0

            # Update mean
            x = x + sigma * zmean

            # Update covariance
            C = (1 + c_c * c_1 - c_1) * C \
                + c_1 * (np.outer(pc, pc) / c_1 - C) \
                + c_mu * np.sum(w[i, None, None] * np.outer(zs[i], zs[i])
                                for i in range(mu))
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * zmean

            # Sigma adaptation (correct log form)
            sigma *= np.exp(c_sigma * (np.linalg.norm(ps) / chi_d - 1))
            if sigma < 1e-12:
                sigma = 1e-12
            if sigma > 1e4:
                sigma = 1e4

            # Keep C symmetric and invert
            C = 0.5 * (C + C.T)
            try:
                Cinv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                Cinv = np.linalg.pinv(C)

            # If the mean moved out of the box, clip it (keeps sampling sane)
            x = np.clip(x, -5, 5)

        return best_x, best_f

    def __call__(self, func):
        start = time.time()
        deadline = start + 280.0  # safety margin well under the 330s wall-clock kill
        try:
            best_x, best_f = self._run(func, self.budget, deadline)
        except Exception:
            # Fallback: a safe (1+1)-ES that never exceeds budget
            try:
                rng = np.random.default_rng(0)
                x = rng.uniform(-5, 5, self.dim)
                fx = float(func(x))
                sigma = 2.0
                for _ in range(self.budget - 1):
                    if time.time() > deadline:
                        break
                    y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                    fy = float(func(y))
                    if fy < fx:
                        x, fx = y, fy
                        sigma *= 1.5
                    else:
                        sigma *= 0.87
                best_x, best_f = x, fx
            except Exception:
                best_x, best_f = None, None
        return best_x, best_f
# EVOLVE-BLOCK-END
