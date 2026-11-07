"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer with adaptive restarts for BBOB Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Deterministic RNG for reproducibility
        rng = np.random.default_rng(12345)

        # Track best solution
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls_used = 1

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))  # population size
        lambda_ = max(lambda_, 8)

        # Initial sigma
        sigma = 1.0

        # Strategy parameters
        cc = (4 + dim / (dim + 4)) / ((dim + 1.3) + 4 + dim / (dim + 4))
        cmu = min(2 - cc + 2 / (dim + 1), 0.5)
        cs = (4 + dim / (dim + 4)) / (dim + 4.1)
        ds = 1 + 2 * max(0, (cc - 1) / (dim + 1)) + cmu

        # State variables
        x_mean = best_x.copy()
        C = np.eye(dim)
        D, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(D, 1e-12))

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Weights
        mu = lambda_ // 2
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Stagnation tracking
        best_f_prev = best_f
        stagnation_count = 0
        no_improve_threshold = 15  # generations without improvement

        # Restart logic
        max_restarts = 3
        restarts = 0

        for gen in range(budget // lambda_ + 1):
            if calls_used >= budget:
                break

            # Sample population
            offspring = []
            for i in range(lambda_):
                if calls_used >= budget:
                    break
                z = rng.standard_normal(dim)
                y = B @ (D * z)
                x_i = np.clip(x_mean + sigma * y, -5, 5)
                offspring.append(x_i)

            # Evaluate offspring
            f_values = []
            for x_i in offspring:
                if calls_used >= budget:
                    break
                f_i = func(x_i)
                f_values.append(f_i)
                calls_used += 1

            if len(f_values) == 0:
                break

            # Sort by fitness
            indices = np.argsort(f_values)
            sorted_x = [offspring[i] for i in indices]
            sorted_f = [f_values[i] for i in indices]

            # Update best
            if sorted_f[0] < best_f:
                best_x = sorted_x[0].copy()
                best_f = sorted_f[0]
                stagnation_count = 0
            else:
                stagnation_count += 1

            # Check for stagnation and restart
            if stagnation_count >= no_improve_threshold and restarts < max_restarts:
                restarts += 1
                # Restart from a new random point
                x_mean = rng.uniform(-5, 5, dim)
                sigma = 1.0
                C = np.eye(dim)
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-12))
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                stagnation_count = 0
                continue

            # Update mean
            x_new_mean = np.sum([weights[i] * sorted_x[i] for i in range(mu)], axis=0)

            # Update sigma
            norm_delta = np.linalg.norm(x_new_mean - x_mean)
            sigma *= np.exp(cs / ds * (norm_delta / (sigma * np.sqrt(mu_eff)) - 1))
            sigma = np.clip(sigma, 1e-8, 100)

            # Update paths
            delta_mean = (x_new_mean - x_mean) / sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * delta_mean

            # h_sigma calculation
            h_sigma = 1.0 if (
                np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))
                < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1))
            ) else 0.0

            p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mu_eff) * delta_mean

            # Update covariance matrix (Rank-1 + Rank-Mu)
            c_mu_p = cmu * (1 - h_sigma * (2 - cc))
            C = (1 + c_mu_p) * C + c_mu_p * np.outer(p_c, p_c)

            for i in range(mu):
                y_i = (sorted_x[i] - x_mean) / sigma
                C += weights[i] * np.outer(y_i, y_i)

            # Symmetrize and ensure PSD
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update mean
            x_mean = x_new_mean

            # Re-decompose C
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-12))

        return best_x, best_f
# EVOLVE-BLOCK-END
