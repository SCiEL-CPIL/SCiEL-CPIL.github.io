"""CMA-ES style optimizer for the BBOB f14 (different powers) task."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive path-based covariance updates.

    A population of 4 points with a full covariance matrix is used so the
    search can align with the rotated axes of the hidden non-separable
    landscape. A deadline guard returns the best point found if the wall-clock
    limit is approached.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 250.0  # safety margin well under 300 s

        # Population size (lambda) = 4, well below budget.
        lam = 4
        mu = lam // 2  # number of recombination weights

        # CMA-ES parameters (standard choices).
        alpha_mu = sum(np.arange(1, mu + 1) ** -2)
        w = (np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))) / alpha_mu
        w = np.clip(w, 0, None)
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (2 + 2.0 / (mu_eff + 1)) / (dim + 2 + 2.0 / (mu_eff + 1))
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 - mu_eff / (dim + 1.7) + c_sigma) / ((dim + 1.7) ** 2 + 2)
        c_mu = np.clip(
            2 * (mu_eff / (dim + 2) - 1) / ((dim + 2) ** 2 + 2) + c_1,
            0, 2 * c_1,
        )
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # Initialization: start near the origin with a large sigma to cover the box.
        rng = np.random.default_rng(12345)
        mean = np.zeros(dim)
        sigma = 4.5
        C = np.eye(dim)
        Cinv = np.eye(dim)
        Csqrt = np.eye(dim)
        Cinv_sqrt = np.eye(dim)
        dcor = 1.0 + 2 * max(0, mu_eff / (dim + 2) - 1) + c_sigma
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        calls = 0

        def _chol_update(C_new):
            nonlocal Csqrt, Cinv_sqrt
            L = np.linalg.cholesky(C_new + 1e-12 * np.eye(dim))
            Csqrt = L
            Cinv_sqrt = np.linalg.inv(L)

        # First evaluation: the mean.
        best_x = mean.copy()
        best_f = func(best_x)
        calls += 1

        # Main loop.
        while calls < budget and time.time() < deadline:
            old_mean = mean.copy()

            # Generate population.
            z = rng.standard_normal((lam, dim))
            pops = old_mean[None, :] + sigma * (z @ Csqrt.T)
            pops = np.clip(pops, -5, 5)

            # Evaluate.
            vals = []
            for i in range(lam):
                if time.time() > deadline:
                    break
                vals.append(func(pops[i]))
                calls += 1
            if len(vals) < lam:
                break
            vals = np.array(vals)

            order = np.argsort(vals)
            sorted_pops = pops[order]
            sorted_vals = vals[order]

            # Update best.
            if sorted_vals[0] < best_f:
                best_f = sorted_vals[0]
                best_x = sorted_pops[0].copy()

            # Mean update (recombination).
            mean = np.sum(w[:, None] * sorted_pops[:mu], axis=0)

            # Path updates (use old_mean).
            y_mean = (mean - old_mean) / (sigma * chi_n)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
            h_sigma = 1.0 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) <
                              (1.4 + 2.0 / (dim + 1)) * chi_n) else 0.0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
            delta_h = (h_sigma - 1.0) * (1 - c_c) * chi_n

            # Covariance matrix update.
            C = ((1 + c_1 + c_mu - c_1 - c_sigma) * C
                 + c_1 * (np.outer(p_c, p_c) + delta_h * np.eye(dim) - C)
                 + c_mu * np.sum(w[:, None, None] * np.outer(sorted_pops[:mu] - old_mean,
                                                              sorted_pops[:mu] - old_mean), axis=0))

            # Sigma adaptation.
            sigma = sigma * np.exp((c_sigma / dcor) *
                                   (np.linalg.norm(p_sigma) / chi_n - 1))

            # Keep sigma in a sane range.
            sigma = float(np.clip(sigma, 1e-10, 50.0))

            # Update Cholesky factors.
            _chol_update(C)

        return best_x, best_f
# EVOLVE-BLOCK-END
