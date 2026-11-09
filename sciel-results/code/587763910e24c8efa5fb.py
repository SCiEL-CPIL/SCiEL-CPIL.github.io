"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with proper Cholesky-based covariance adaptation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(12345)
        dim = self.dim

        # CMA-ES parameters
        lambda_pop = max(4 + int(3 * np.log(dim)), 10)  # population size
        mu = lambda_pop // 2  # number of parents
        # Weights (logarithmic)
        w = np.log(np.arange(mu, 0, -1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        # Step size
        sigma = 1.0
        cc = (4 + mu_eff / dim) / (dim + 4.7 + mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = (2 - 1 / mu_eff + 2 / dim) / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs

        # Initial center
        x_mean = rng.uniform(-3, 3, dim)
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        C_sqrt = np.eye(dim)

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        def _update_chol(C, C_inv_sqrt, C_sqrt):
            """Update Cholesky factorization using eigendecomposition for stability."""
            # Use eigendecomposition for numerical stability
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip eigenvalues to be positive
            eigvals = np.clip(eigvals, 1e-10, None)
            C_inv_sqrt = (eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T)
            C_sqrt = (eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T)
            # Symmetrize
            C_inv_sqrt = (C_inv_sqrt + C_inv_sqrt.T) / 2
            C_sqrt = (C_sqrt + C_sqrt.T) / 2
            return C_inv_sqrt, C_sqrt

        # Initial Cholesky
        C_inv_sqrt, C_sqrt = _update_chol(C, C_inv_sqrt, C_sqrt)

        best_x = x_mean.copy()
        best_f = func(best_x)
        calls_made = 1

        # Track best value after each call for potential use
        for gen in range(self.budget // lambda_pop + 1):
            if time.time() > deadline:
                break
            if calls_made >= self.budget:
                break

            # Generate population
            pop = []
            for i in range(lambda_pop):
                if time.time() > deadline or calls_made >= self.budget:
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                pop.append((fy, y))

            if not pop:
                break

            # Sort by fitness
            pop.sort(key=lambda t: t[0])

            # Update best
            if pop[0][0] < best_f:
                best_f = pop[0][0]
                best_x = pop[0][1].copy()

            # Compute weighted mean of best mu points
            y_w = np.zeros(dim)
            for i in range(min(mu, len(pop))):
                y_w += w[i] * pop[i][1]
            x_mean_new = y_w

            # Update evolution path for sigma
            C_sqrt_old = C_sqrt.copy()
            z_w = np.zeros(dim)
            for i in range(min(mu, len(pop))):
                z_i = np.linalg.solve(C_sqrt_old, pop[i][1] - x_mean)
                z_w += w[i] * z_i
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * z_w

            # Update evolution path for covariance
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mu_eff) * (x_mean_new - x_mean) / sigma

            # Update covariance matrix
            C = ((1 + c1 * (2 - c1)) * C - c1 * np.outer(p_c, p_c) +
                 cmu * sum(w[i] * np.outer(pop[i][1] - x_mean, pop[i][1] - x_mean)
                           for i in range(min(mu, len(pop)))))

            # Ensure positive definiteness
            C = (C + C.T) / 2
            eigvals, _ = np.linalg.eigh(C)
            if np.min(eigvals) < 1e-10:
                C += 1e-8 * np.eye(dim)

            x_mean = x_mean_new

            # Update sigma
            p_sigma_norm = np.linalg.norm(p_sigma)
            chisq = np.sqrt(mu_eff) * (1 + 1 / (2 * mu_eff) - 1 / (8 * mu_eff ** 2))
            sigma *= np.exp(cs / damps * (p_sigma_norm / chisq - 1))
            sigma = np.clip(sigma, 1e-10, 50)

            # Update Cholesky factors
            C_inv_sqrt, C_sqrt = _update_chol(C, C_inv_sqrt, C_sqrt)

        return best_x, best_f
# EVOLVE-BLOCK-END
