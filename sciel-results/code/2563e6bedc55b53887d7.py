"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style population-based optimizer with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s kill
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Population size
        lambda_ = int(4 + 3 * np.log(n))
        weights = np.log(lambda_ + 0.5) - np.log(np.arange(1, lambda_ + 1))
        weights /= weights.sum()
        mu = int(np.sum(weights > 0))
        weights = weights[:mu]
        weights /= weights.sum()
        c_mu = 2 / ((n + 1.3) ** 2 + mu)

        # Step size
        sigma = 1.0

        # Initial mean
        m = rng.uniform(-4, 4, n)
        C = np.eye(n)
        C_inv = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)

        # Update parameters
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        c_sigma = (mu + 0.5) / (n + mu + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + n / c_sigma) / (n + 4 + 2 * c_sigma)
        c_c = (4 + n / c_sigma) / (n + 4 + 2 * c_sigma)
        c_1 = c_mu / (n ** 2 + 2)
        c_mu = (2 * (c_mu - 1) + c_mu ** 2) / (n ** 2 + 2)
        c_mu = min(c_mu, 2 * (1 - c_1) / n)

        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        # Evaluate initial mean
        m_clipped = np.clip(m, -5, 5)
        f_m = func(m_clipped)
        best_x = m_clipped.copy()
        best_f = f_m

        calls_made = 1
        iterations = 0

        while calls_made < budget:
            if time.time() > deadline:
                break
            iterations += 1

            # Generate population
            y_list = []
            for i in range(lambda_):
                z = rng.standard_normal(n)
                # Transform: m + sigma * B @ (D * z)
                y = m + sigma * (B @ (D * z))
                y_list.append(y)

            # Evaluate population
            f_y_list = []
            for y in y_list:
                y_clipped = np.clip(y, -5, 5)
                f_y = func(y_clipped)
                f_y_list.append(f_y)
                calls_made += 1
                if f_y < best_f:
                    best_f = f_y
                    best_x = y_clipped.copy()

            if calls_made >= budget:
                break

            # Sort by fitness
            indices = np.argsort(f_y_list)
            y_selected = [y_list[i] for i in indices[:mu]]
            f_selected = [f_y_list[i] for i in indices[:mu]]

            # Compute weighted mean
            m_new = np.sum(np.array(weights)[:, None] * np.array(y_selected), axis=0)

            # Check convergence
            if np.linalg.norm(m_new - m) / max(sigma, 1e-10) < 1e-10:
                break

            # Update p_sigma
            y_w = np.sum(np.array(weights)[:, None] * np.array(y_selected), axis=0)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (
                (m_new - m) / (sigma * np.sqrt(np.trace(C)))
            )
            mu_eff = 1 / np.sum(weights ** 2)

            # Update p_c
            h_sig = (
                np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (iterations + 1)))
                < (1 + 2 / (n + 1)) * chi_n
            )
            p_c = (1 - c_c) * p_c + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (
                (m_new - m) / sigma
            )

            # Update C
            delta_h = (h_sig - 1) * c_c * (2 - c_c)
            C = (
                (1 - c_1 - c_mu + delta_h) * C
                + c_1 * (np.outer(p_c, p_c) + (1 - h_sig) * c_c * (2 - c_c) * C)
                + c_mu * np.sum(weights[i] * np.outer(y_selected[i] - m, y_selected[i] - m) for i in range(mu))
            )

            # Eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            D = np.sqrt(eigvals)
            B = eigvecs

            # Update sigma
            sigma *= np.exp(
                c_sigma / d_sigma
                * (np.linalg.norm(p_sigma) / chi_n - 1)
            )

            # Update mean
            m = m_new

            # Safety: keep sigma reasonable
            sigma = np.clip(sigma, 1e-10, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
