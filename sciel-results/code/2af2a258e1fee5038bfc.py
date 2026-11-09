"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time-aware budget management for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 330s kill - 40s safety margin

        rng = np.random.default_rng(0)
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = 16
        mu = lambda_ // 2
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)

        # Step size control
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 1) - 1) + c_sigma
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Covariance matrix adaptation
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = (2 * (mu_eff - 2 + 2 / mu_eff)) / ((n + 2) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        damping = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1) - 1)) + c_mu

        # Initialization
        x = rng.uniform(-4, 4, n)  # Start near center, within [-4,4]
        sigma = 1.5
        C = np.eye(n)
        C_sqrt = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        calls_used = 0

        def evaluate(x_vec):
            nonlocal calls_used
            x_clipped = np.clip(x_vec, -5, 5)
            f_val = func(x_clipped)
            calls_used += 1
            return x_clipped, f_val

        # Initial evaluation
        x, fx = evaluate(x)
        best_x, best_fx = x.copy(), fx

        # Main CMA-ES loop
        while calls_used < budget:
            if time.time() >= deadline:
                break

            # Generate offspring
            offspring = []
            offspring_fx = []
            for i in range(lambda_):
                if time.time() >= deadline:
                    break
                # Sample from N(x, sigma^2 * C)
                z = rng.standard_normal(n)
                # Apply C_sqrt
                y = x + sigma * (C_sqrt @ z)
                y_clipped, fy = evaluate(y)
                offspring.append(y_clipped)
                offspring_fx.append(fy)

            if len(offspring) < 2:
                break

            # Sort by fitness
            indices = np.argsort(offspring_fx)
            offspring = [offspring[i] for i in indices]
            offspring_fx = [offspring_fx[i] for i in indices]

            # Update best
            if offspring_fx[0] < best_fx:
                best_fx = offspring_fx[0]
                best_x = offspring[0].copy()

            # Mean update
            new_x = sum(weights[i] * offspring[i] for i in range(mu))

            # Step size control
            diff = (new_x - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff / chi_n
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (n + 1)) * chi_n:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Covariance matrix adaptation
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_x - x) / sigma
            delta_heaviside = 1 if np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** 2) < (1.4 + 2 / (n + 1)) * chi_n else 0

            rank_mu = np.zeros((n, n))
            for i in range(mu):
                diff_i = (offspring[i] - x) / sigma
                rank_mu += weights[i] * np.outer(diff_i, diff_i)

            C = ((1 + c_c * (delta_heaviside - 1) - c_1 - c_mu) * C
                 + c_c * (np.outer(p_c, p_c) + (delta_heaviside - 1) * C)
                 + c_1 * np.eye(n)
                 + c_mu * rank_mu)

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback: regularize
                C += 1e-6 * np.eye(n)
                C = (C + C.T) / 2
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C_sqrt = np.eye(n)

            x = new_x

        return best_x, best_fx
# EVOLVE-BLOCK-END
