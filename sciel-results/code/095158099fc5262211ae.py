"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with anisotropic covariance adaptation for ill-conditioned landscapes."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # 15s safety margin before 300s limit

        # CMA-ES parameters
        mu = max(1, dim // 2)  # number of recombination weights
        lambda_ = 2 * mu       # population size
        # recombination weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        # step size control
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # evolution path for covariance matrix adaptation
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # initial population
        rng = np.random.default_rng(42)
        x = np.zeros(dim)  # start at origin, shift will be handled by the function
        sigma = 2.0

        # initial covariance matrix
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)

        # evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # cache for Cholesky
        def update_chol(C):
            try:
                L = np.linalg.cholesky(C)
                return L, np.linalg.inv(L.T)
            except np.linalg.LinAlgError:
                # fallback if C is not positive definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                L = eigvecs @ np.diag(np.sqrt(eigvals))
                C_inv_sqrt = np.linalg.inv(L.T)
                return L, C_inv_sqrt

        C_sqrt, C_inv_sqrt = update_chol(C)

        # track best
        best_x = x.copy()
        best_f = func(x)
        calls = 1

        for gen in range(budget):
            # check time
            if time.time() > deadline:
                break

            # generate population
            ys = np.zeros((lambda_, dim))
            fs = np.zeros(lambda_)
            for i in range(lambda_):
                if time.time() > deadline:
                    break
                y = x + sigma * C_sqrt @ rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                ys[i] = y
                fs[i] = func(y)
                calls += 1
                if fs[i] < best_f:
                    best_f = fs[i]
                    best_x = y.copy()

            # sort by fitness
            order = np.argsort(fs)
            ys_sorted = ys[order]
            fs_sorted = fs[order]

            # update mean
            x_new = np.sum(w[:, None] * ys_sorted[:mu], axis=0)
            x_new = np.clip(x_new, -5, 5)

            # update sigma
            y_sigma = (C_inv_sqrt @ (x_new - x)) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 1e12)

            # update covariance
            y_c = (C_inv_sqrt @ (x_new - x)) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
            delta_sigma = (1 - c_c) ** 2 - 1
            C = ((1 + c_1 + c_mu - c_c) * C
                 + c_1 * (np.outer(y_c, y_c) - np.eye(dim))
                 + c_mu * (np.outer(y_c, y_c)
                          - delta_sigma * C
                          - np.sum(w[:, None] * np.outer(ys_sorted[:mu] - x, ys_sorted[:mu] - x), axis=0) / sigma ** 2 * C_inv_sqrt @ C @ C_inv_sqrt.T))
            # Simplified: just use the standard CMA-ES update
            C = (1 + c_1 + c_mu - c_c) * C
            C += c_1 * (np.outer(y_c, y_c) - C)
            C += c_mu * (np.sum(w[:, None, None] * np.outer(ys_sorted[:mu] - x, ys_sorted[:mu] - x), axis=0) / sigma ** 2 - C)

            # ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            C_sqrt, C_inv_sqrt = update_chol(C)

            x = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
