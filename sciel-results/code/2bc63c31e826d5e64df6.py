"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        mu = max(1, n // 2)  # number of parents
        lambda_ = 2 * mu      # population size
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma

        # Initial point: center of the box
        x = np.zeros(n)
        sigma = 1.0
        C = np.eye(n)
        C_sqrt_inv = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Helper to compute matrix square root inverse via eigen decomposition
        def update_C_sqrt_inv(C):
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            return C_sqrt, C_sqrt_inv

        C_sqrt, C_sqrt_inv = update_C_sqrt_inv(C)

        best_x = x.copy()
        best_f = func(x)
        calls_used = 1

        # Pre-generate random numbers if needed? No, use rng on the fly.
        rng = np.random.default_rng(42)

        while calls_used < budget:
            # Generate population
            Y = []
            F = []
            for i in range(lambda_):
                z = rng.standard_normal(n)
                y = x + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1
                Y.append(y)
                F.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            # Sort by fitness
            idx = np.argsort(F)
            Y = [Y[i] for i in idx]
            F = [F[i] for i in idx]

            # Update mean
            x_old = x.copy()
            x = np.sum([weights[i] * Y[i] for i in range(mu)], axis=0)

            # Update step size control variable
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ (x - x_old) / sigma)

            # Update evolution path for covariance matrix
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * max(1, calls_used // lambda_)))) < (1.4 + 2 / (n + 1)) * chi_n
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x - x_old) / sigma

            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C
            C += c_1 * np.outer(p_c, p_c)
            C += c_mu * np.sum([weights[i] * np.outer(C_sqrt_inv @ (Y[i] - x_old), C_sqrt_inv @ (Y[i] - x_old)) for i in range(mu)], axis=0)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Update C_sqrt_inv
            C_sqrt, C_sqrt_inv = update_C_sqrt_inv(C)

            # If sigma is very small or we've converged, break early?
            # The task says "More than 500 calls... makes the whole result invalid".
            # It also says "repeating the last n_i if fewer calls are made".
            # So we can stop early if we think we found the optimum.
            # However, for BBOB f4, it's a landscape with many local minima.
            # CMA-ES is global.
            # Let's just run until budget is exhausted to be safe, unless sigma is tiny.
            if sigma < 1e-6:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
