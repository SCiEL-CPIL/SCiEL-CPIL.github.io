"""Rewritten optimizer for BBOB f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time guard for BBOB f13."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        # Initial point
        x = np.zeros(d)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # CMA-ES parameters
        sigma = 2.0
        lam = max(4, int(4 + 3 * np.log(d)))
        mu = lam // 2

        C = np.eye(d)
        C_inv = np.eye(d)
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        chi_n = np.sqrt(d) * (1 - 1.0 / (4 * d) + 1.0 / (21 * d ** 2))

        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        nu = weights.sum() ** 2 / (weights ** 2).sum()
        c_sigma = (nu + 2) / (d + nu + 5)
        c_c = (4 + nu / d) / (d + 4 + 2 * nu / d)
        c_1 = 2 / ((d + 1.3) ** 2 + nu)
        c_mu = min(1 - c_1, 2 * (nu - 2 + 1 / d) / ((d + 2) ** 2 + nu))

        while calls < budget:
            # Time check
            if time.time() > deadline:
                break

            # Sample offspring
            offspring = []
            for i in range(lam):
                if time.time() > deadline:
                    break
                z = rng.standard_normal(d)
                try:
                    L = np.linalg.cholesky(C)
                    y = x + sigma * (L @ z)
                except np.linalg.LinAlgError:
                    y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                offspring.append((fy, y))

            if len(offspring) < mu:
                break

            offspring.sort(key=lambda t: t[0])

            # Update best
            if offspring[0][0] < best_fx:
                best_fx = offspring[0][0]
                best_x = offspring[0][1].copy()

            # Weighted mean
            y_mean = np.zeros(d)
            for i in range(mu):
                y_mean += weights[i] * offspring[i][1]

            # Compute C^{-1/2}
            try:
                C_inv_chol = np.linalg.cholesky(C_inv)
                C_inv_sqrt = C_inv_chol.T @ np.linalg.inv(C_inv_chol)
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(d)

            # Update paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * nu) * C_inv_sqrt @ (y_mean - x) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * nu) * C_inv_sqrt @ (y_mean - x)

            h_sig = (np.linalg.norm(p_sigma) / chi_n) < (1.4 + 2 / (d + 1))

            # Update C
            delta_h = (1.8 + 5 / (d + mu - 2)) * (1 - h_sig) - 1
            rank_1 = np.outer(p_c, p_c)
            rank_mu = np.zeros((d, d))
            for i in range(mu):
                zi = C_inv_sqrt @ (offspring[i][1] - y_mean) / sigma
                rank_mu += weights[i] * np.outer(zi, zi)

            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank_1 + c_mu * rank_mu
            C = 0.5 * (C + C.T)  # Symmetrize

            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + 1e-8 * np.eye(d)

            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            x = y_mean

            try:
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                C_inv = np.eye(d)

        return best_x, best_fx
# EVOLVE-BLOCK-END
