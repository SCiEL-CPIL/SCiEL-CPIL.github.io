"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES implementation for BBOB f6 (Attractive sector)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0

        rng = np.random.default_rng(12345)
        dim = self.dim

        # Initialize
        x = np.zeros(dim)
        f_x = func(x)
        best_x = x.copy()
        best_f = f_x
        calls = 1

        sigma = 2.0
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)

        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(dim)))
        mu = lam // 2
        w = np.log(np.arange(1, mu + 1)) - np.log(mu)
        w = w / w.sum()
        w = np.concatenate([w, np.zeros(lam - mu)])
        mu_eff = 1 / np.sum(w[:mu] ** 2)

        c_sigma = (lam + 2) / (dim + lam + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((lam + 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
        c_1 = (2 + 2 / (dim + 1.3)) / ((dim + 2) + 2 / (dim + 1.3))
        c_mu = min(1 - c_1, (2 * (mu_eff / lam - 1) + c_1) /
                   ((dim + 2) + 2 * mu_eff / lam))

        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        while calls < self.budget:
            if time.time() > deadline - 0.5:
                break

            # Generate population
            pop = []
            fvals = []
            for i in range(lam):
                if time.time() > deadline - 0.1:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                f_y = func(y)
                pop.append(y)
                fvals.append(f_y)
                calls += 1
                if f_y < best_f:
                    best_f = f_y
                    best_x = y.copy()

            if len(pop) < 2:
                break

            actual_lam = len(pop)
            if actual_lam < lam:
                lam = actual_lam
                mu = max(1, lam // 2)
                w = np.log(np.arange(1, mu + 1)) - np.log(mu)
                w = w / w.sum()
                w = np.concatenate([w, np.zeros(lam - mu)])
                mu_eff = 1 / np.sum(w[:mu] ** 2)
                c_mu = min(1 - c_1, (2 * (mu_eff / lam - 1) + c_1) /
                           ((dim + 2) + 2 * mu_eff / lam))

            # Sort by fitness
            idx = np.argsort(fvals)
            pop = [pop[i] for i in idx]
            fvals = [fvals[i] for i in idx]

            # Update mean
            x_old = x.copy()
            x_new = x_old.copy()
            for i in range(min(mu, lam)):
                x_new += w[i] * (pop[i] - x_old)
            x = np.clip(x_new, -5, 5)

            # Compute weighted sum of step vectors
            y_w = np.array([(pop[i] - x_old) / sigma for i in range(min(mu, lam))])
            y_w = y_w.T  # shape: (dim, mu)

            # Update evolution path for step size control
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * \
                 (B @ (D * np.mean(y_w, axis=1)))

            # Update step size
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Update evolution path for covariance
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * \
                 (B @ (D * np.mean(y_w, axis=1)))

            # Update covariance matrix
            delta_h = 1 if np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** (2 * (calls - lam))) < \
                        (1.4 + 2 / (dim + 1)) * chi_n else 0
            C = (1 + c_1 * (delta_h - 1)) * C + \
                c_1 * (1 - delta_h) * np.outer(pc, pc) + \
                c_mu / sigma ** 2 * (y_w @ np.diag(w[:min(mu, lam)]) @ y_w.T)

            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            C = (eigvecs * eigvals) @ eigvecs.T
            D = np.sqrt(eigvals)
            B = eigvecs

        return best_x, best_f
# EVOLVE-BLOCK-END
