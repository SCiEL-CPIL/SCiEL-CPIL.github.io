"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for Rastrigin f3."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size
        lam = 4 + int(3 * np.log(n))
        # Selection size
        mu = lam // 2

        # CMA-ES parameters
        c_sigma = (mu_eff + 2) / ((n + 1.3) ** 2 + mu_eff)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, (mu_eff - 2 + 2 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Initial population
        x0 = np.zeros(n)
        sigma = 2.0

        # Covariance matrix
        C = np.eye(n)
        # Square root of C
        C_sqrt = np.eye(n)
        # Inverse square root of C
        C_inv_sqrt = np.eye(n)

        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        # Weights for recombination
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        # Best solution tracking
        best_x = x0.copy()
        best_f = func(best_x)
        calls_made = 1

        # Main loop
        while calls_made < budget:
            # Generate population
            offspring = np.zeros((lam, n))
            for i in range(lam):
                z = rng.standard_normal(n)
                y = x0 + sigma * (C_sqrt @ z)
                offspring[i] = np.clip(y, -5, 5)

            # Evaluate offspring
            fvals = np.array([func(offspring[i]) for i in range(lam)])
            calls_made += lam

            # Update best if any offspring is better
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = offspring[min_idx].copy()

            if calls_made >= budget:
                break

            # Sort by fitness
            idx = np.argsort(fvals)
            x_sorted = offspring[idx]
            f_sorted = fvals[idx]

            # Update mean
            x_old = x0.copy()
            x0 = sum(w[i] * x_sorted[i] for i in range(mu))

            # Update sigma
            y_old = (x0 - x_old) / (sigma * chi_n) if sigma > 0 else np.zeros(n)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_old
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (calls_made // lam + 1)) <
                       (1.4 + 2 / (n + 1)) * chi_n)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 10.0)

            # Update C
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_old
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + \
                c_1 * (np.outer(p_c, p_c) + delta_h * np.eye(n)) + \
                c_mu * sum(w[i] * np.outer(y_sorted[i] - x_old, y_sorted[i] - x_old) / sigma ** 2
                           for i in range(mu)) if sigma > 0 else C

            # Ensure symmetry
            C = 0.5 * (C + C.T)

            # Eigen decomposition for numerical stability
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C = 0.5 * (C + C.T)

            # Update square roots
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals))

            # Clip mean to box
            x0 = np.clip(x0, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
