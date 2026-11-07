"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        n = self.budget

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(d)))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = (2 - mu_eff / (d + 1.5) + c_sigma) / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(d) * (1 - 1.0 / (4 * d) + 1.0 / (21 * d ** 2))

        # Initial state
        x_mean = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        C_eigh = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(C_eigh[0], 1e-20))
        B = C_eigh[1]
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)

        best_x = x_mean.copy()
        best_f = func(best_x)
        calls = 1

        while calls < n:
            # Generate population
            pop = []
            for i in range(lambda_):
                if calls >= n:
                    break
                z = np.random.randn(d)
                x_i = x_mean + sigma * (B @ (D * z))
                x_i = np.clip(x_i, -5, 5)
                f_i = func(x_i)
                calls += 1
                pop.append((f_i, x_i, z))

            if len(pop) < mu:
                break

            # Sort by fitness
            pop.sort(key=lambda t: t[0])

            # Update best
            if pop[0][0] < best_f:
                best_f = pop[0][0]
                best_x = pop[0][1].copy()

            # Compute weighted mean of selected points
            y_w = np.array([pop[i][1] - x_mean for i in range(mu)])
            x_mean_new = x_mean + sum(weights[i] * y_w[i] for i in range(mu))

            # Update step-size control
            weighted_z = sum(weights[i] * pop[i][2] for i in range(mu))
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D * weighted_z))

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e5)

            # Update path for covariance
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (B @ (D * weighted_z))

            # Update covariance matrix
            C_old = C.copy()
            C = (1 - c_1 - c_mu) * C_old
            C += c_1 * (np.outer(p_c, p_c) + (c_c * (2 - c_c) - 1) * C_old)
            for i in range(mu):
                C += c_mu * weights[i] * np.outer(y_w[i], y_w[i])

            # Ensure symmetry
            C = (C + C.T) / 2

            # Update B and D via eigendecomposition
            C_eigh = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(C_eigh[0], 1e-20))
            B = C_eigh[1]

            x_mean = x_mean_new

        return best_x, best_f
# EVOLVE-BLOCK-END
