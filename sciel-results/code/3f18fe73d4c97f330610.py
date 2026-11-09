"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer for the Bent Cigar function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget

        # Initial population size
        lambda_ = max(4, int(4 + 3 * np.log(d)))
        
        # Step size adaptation parameters
        c_sigma = (lambda_ + 2) / (d + lambda_ + 5)
        d_sigma = 1.0 + 2 * max(0, np.sqrt((lambda_ - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + lambda_ / d) / (d + 4 + 2 * lambda_ / d)
        c_1 = 2 / ((d + 1.3) ** 2 + lambda_)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 0.5) / ((d + 2) ** 2 + mu_eff)))
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        chi_n = np.sqrt(d) * (1 - 1.0 / (4 * d) + 1.0 / (21 * d ** 2))

        # State variables
        x = np.zeros(d)
        sigma = 2.0
        ps = np.zeros(d)
        pc = np.zeros(d)
        B = np.eye(d)
        D = 1.0
        D_inv = 1.0
        alpha_sigma = 1.0
        cs = 1.0

        best_x = x.copy()
        best_f = float('inf')
        calls = 0

        def evaluate(x_vec):
            nonlocal calls
            f_val = float(func(x_vec))
            calls += 1
            return f_val

        # Initial evaluation
        best_f = evaluate(x)
        best_x = x.copy()

        # Pre-compute constants
        c_sigma = (lambda_ + 2) / (d + lambda_ + 5)
        d_sigma = 1.0 + 2 * max(0, np.sqrt((lambda_ - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + lambda_ / d) / (d + 4 + 2 * lambda_ / d)
        c_1 = 2 / ((d + 1.3) ** 2 + lambda_)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 0.5) / ((d + 2) ** 2 + mu_eff)))

        while calls < budget:
            # Generate offspring
            offspring = []
            for i in range(lambda_):
                if calls >= budget:
                    break
                z = np.random.standard_normal(d)
                y = x + sigma * (D * B @ z)
                y = np.clip(y, -5, 5)
                fy = evaluate(y)
                offspring.append((fy, y))

            if not offspring:
                break

            # Sort offspring
            offspring.sort(key=lambda t: t[0])
            y_sorted = [off[i][1] for off in offspring]
            f_sorted = [off[i][0] for off in offspring]

            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = y_sorted[0].copy()

            # Weighted mean of mu best offspring
            y_w = np.zeros(d)
            for i in range(mu):
                y_w += weights[i] * y_sorted[i]

            # Update path evolution strategies
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D_inv * (y_w - x)))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget - calls)))) < (1.4 + 2 / (d + 1)) * chi_n
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (y_w - x)

            # Update covariance matrix
            delta_d = (1 - c_1 - c_mu) if hsig else 1.0
            rank_mu = np.outer(y_w - x, y_w - x)
            rank_one = np.outer(pc, pc)
            C_new = (1 + c_1 * delta_d - c_1 - c_mu) * B @ B.T + c_1 * rank_one + c_mu * rank_mu
            C_new = (C_new + C_new.T) / 2

            # Eigendecomposition
            eigenvalues, B_new = np.linalg.eigh(C_new)
            eigenvalues = np.clip(eigenvalues, 1e-20, None)
            D_new = np.sqrt(eigenvalues)
            D_inv_new = 1.0 / D_new
            B = B_new
            D = D_new
            D_inv = D_inv_new

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Update mean
            x = y_w

            # Handle boundary issues by projecting back if needed
            x = np.clip(x, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
