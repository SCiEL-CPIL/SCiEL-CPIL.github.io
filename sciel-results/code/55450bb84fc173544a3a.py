"""CMA-ES optimizer for BBOB f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive step-size for box-constrained optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # 280s safety margin out of 300s

        rng = np.random.default_rng(42)

        # Initial setup
        x = rng.uniform(-5, 5, dim)
        sigma = 2.0
        C = np.eye(dim)
        Cinv = np.eye(dim)

        # CMA-ES parameters
        mu = max(1, dim // 2)
        lambda_pop = 2 * mu
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        call_count = 1

        def _sqrtm(C_mat):
            w, V = np.linalg.eigh(C_mat)
            w = np.clip(w, 1e-20, None)
            return V @ np.diag(np.sqrt(w)) @ V.T

        Csqrt = _sqrtm(C)
        generation = 0

        while call_count < budget:
            if time.time() > deadline:
                break

            ys = np.zeros((lambda_pop, dim))
            fs = np.zeros(lambda_pop)
            for i in range(lambda_pop):
                if time.time() > deadline or call_count >= budget:
                    break
                y = x + sigma * (Csqrt @ rng.standard_normal(dim))
                y = np.clip(y, -5, 5)
                ys[i] = y
                fs[i] = func(y)
                call_count += 1
                if fs[i] < best_fx:
                    best_fx = fs[i]
                    best_x = y.copy()

            # Only update if we have at least mu valid evaluations
            valid = np.isfinite(fs)
            if valid.sum() < mu:
                break

            order = np.argsort(fs)
            ys_sorted = ys[order]
            fs_sorted = fs[order]

            if fs_sorted[0] < best_fx:
                best_fx = fs_sorted[0]
                best_x = ys_sorted[0].copy()

            x_old = x.copy()
            x = np.sum(weights[:, None] * ys_sorted[:mu], axis=0)
            x = np.clip(x, -5, 5)

            y_sigma = (Cinv @ (x - x_old)) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma = (norm_p_sigma / np.sqrt(1 - (1 - c_sigma) ** (2 * (generation + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            sigma = max(sigma, 1e-12)
            sigma = min(sigma, 10.0)

            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x - x_old) / sigma

            if h_sigma:
                C += (1 + (1 - h_sigma) * c_c * (2 - c_c) - c_1 - c_mu) * np.outer(p_c, p_c)
            else:
                C += (1 - c_1 - c_mu) * np.outer(p_c, p_c)

            H = np.zeros((dim, dim))
            for i in range(mu):
                diff = (ys_sorted[i] - x_old) / sigma
                H += weights[i] * np.outer(diff, diff)
            C += c_mu * H

            try:
                w, V = np.linalg.eigh(C)
                w = np.clip(w, 1e-10, None)
                C = V @ np.diag(w) @ V.T
                Cinv = V @ np.diag(1.0 / w) @ V.T
                Csqrt = V @ np.diag(np.sqrt(w)) @ V.T
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                Cinv = np.eye(dim)
                Csqrt = np.eye(dim)

            generation += 1

        return best_x, best_fx
# EVOLVE-BLOCK-END
