"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        mu = max(1, n // 2)  # number of parents
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step size
        sigma = 2.0

        # Initial point
        rng = np.random.default_rng(42)
        x = rng.uniform(-5, 5, n)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        # Covariance matrix
        C = np.eye(n)
        # Inverse square root of C (initially identity)
        C_inv_sqrt = np.eye(n)
        # Eigen decomposition for updating
        D = np.ones(n)  # eigenvalues of C
        B = np.eye(n)   # eigenvectors of C

        # Learning rates
        c_sigma = (mu_eff + 2) / (n + 4)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / (n + 4))
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # State variables
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        hs = 1.0

        calls_made = 1
        for _ in range(budget - 1):
            # Generate offspring
            offspring = []
            for i in range(mu):
                z = rng.standard_normal(n)
                # Transform: B * D * z
                y = x + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                offspring.append(y)

            # Evaluate offspring
            offspring_f = []
            for y in offspring:
                fy = func(y)
                calls_made += 1
                offspring_f.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            # Sort offspring by fitness
            indices = np.argsort(offspring_f)
            sorted_offspring = [offspring[i] for i in indices]
            sorted_f = [offspring_f[i] for i in indices]

            # Update center of mass
            x_new = np.sum([weights[i] * sorted_offspring[i] for i in range(mu)], axis=0)

            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ ((x_new - x) / sigma)

            # Hessian potential
            if np.linalg.norm(p_sigma) < (1.4 + 2 / (n + 1)) * chi_n * hs:
                hs = (1 - c_sigma) * hs + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff)

            # Adapt sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Update p_c
            h_sigma = 1.0 if np.linalg.norm(p_sigma) < (1.4 + 2 / (n + 1)) * chi_n * hs else 0.0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt @ ((x_new - x) / sigma)

            # Update C
            delta_h = (h_sigma - 1) * c_c * (2 - c_c)
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                z_i = C_inv_sqrt @ ((sorted_offspring[i] - x) / sigma)
                rank_mu += weights[i] * np.outer(z_i, z_i)

            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

            # Update B and D (eigen decomposition)
            # Periodically do full eigen decomposition
            # For efficiency, we can do it every few iterations or use a simpler approach
            # Let's do it every 10 iterations for stability
            # Actually, for n=5, it's cheap enough to do every iteration
            try:
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-16))
                C_inv_sqrt = (B @ np.diag(1.0 / D)) @ B.T
            except:
                # Fallback if eigen decomposition fails
                C_inv_sqrt = np.eye(n)

            # Update x
            x = x_new
            fx = sorted_f[0]

            # Optional: restart if stuck (not needed for 500 budget)
            # But let's add a simple stagnation check
            if sigma < 1e-8:
                sigma = 1.0

        return best_x, best_f
# EVOLVE-BLOCK-END
