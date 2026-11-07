"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES based optimizer for the BBOB-style f17 (Schaffers F7) function.
    Uses a fixed-seed CMA-ES with careful budget management and time guard.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin before 330s kill

        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        # Initial population size
        lambda_ = max(4, int(4 + 3 * np.log(d)))
        mu = lambda_ // 2

        # Initial point: center of the box
        x = np.zeros(d)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Initial step size
        sigma = 1.0

        # CMA-ES parameters
        alpha_mu = sum(range(1, mu + 1)) / mu
        c_sigma = (mu + 2) / (d + mu + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu / d) / (d + 4 + 2 * mu / d)
        c_1 = (2 / ((d + 1.3) ** 2 + mu)) * (1 - alpha_mu)
        c_mu = (2 / ((d + 2) ** 2 + mu)) * (min(1 - alpha_mu, 2 * alpha_mu) - 1)

        # Evolution paths
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)

        # Covariance matrix
        C = np.eye(d)
        C_eigenvalues = np.ones(d)
        C_eigenvectors = np.eye(d)

        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)

        def sample_from_C(sigma):
            """Sample lambda points from N(x, sigma^2 * C)"""
            # Use eigendecomposition for sampling
            z = rng.standard_normal((lambda_, d))
            # Project through eigenvectors and scale by sqrt(eigenvalues)
            y = (z * np.sqrt(C_eigenvalues)) @ C_eigenvectors.T
            return x + sigma * y

        def update_CMAES(y_sorted, f_sorted):
            """Update CMA-ES parameters after selecting the best mu points"""
            nonlocal p_sigma, p_c, C, C_eigenvalues, C_eigenvectors, sigma

            # Weighted mean of selected points
            m = np.sum(weights[:, None] * y_sorted[:mu], axis=0)

            # Update evolution paths
            # Compute inverse square root of C for the p_sigma update
            # Use eigendecomposition
            C_inv_sqrt = (C_eigenvectors * (1.0 / np.sqrt(C_eigenvalues))) @ C_eigenvectors.T

            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ m / sigma

            # Update p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * m / sigma

            # Update covariance matrix
            # Rank-1 update
            delta_sigma = np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (p_sigma / sigma - (1 - c_sigma) * p_sigma)
            # Actually, the standard rank-1 update uses the weighted mean
            rank1 = np.outer(m, m)

            # Rank-mu update
            rankmu = np.sum(weights[:, None] * np.outer(y_sorted[:mu], y_sorted[:mu]), axis=0)

            # Correction term for the rank-1 update
            h_sigma = (
                np.sum(p_sigma ** 2) / (1 - (1 - c_sigma) ** (2 * calls))
            )
            h_sigma = 1 if h_sigma < (1 + 4 / (d + 1)) * (1 - (1 - c_sigma) ** 2) else 0

            C = (
                (1 + c_1 + c_mu - c_1 * d) * C
                + c_1 * (np.outer(p_c, p_c) / (1 - (1 - c_c) ** 2) - (1 - h_sigma) * c_sigma * (2 - c_sigma) * C)
                + c_mu * (rankmu - C)
            )

            # Ensure symmetry
            C = (C + C.T) / 2

            # Eigen decomposition (simple, since d=5)
            C_eigenvalues, C_eigenvectors = np.linalg.eigh(C)
            # Clip eigenvalues to be positive
            C_eigenvalues = np.maximum(C_eigenvalues, 1e-20)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(2 * d) - 1))

            # Update x
            x = x + m

        # Main loop
        mu_eff = 1.0 / np.sum(weights ** 2)

        while calls < budget:
            # Check time
            if time.time() > deadline:
                break

            # Sample population
            population = sample_from_C(sigma)

            # Evaluate all points in the population
            f_values = []
            for i in range(lambda_):
                if time.time() > deadline:
                    break
                x_i = np.clip(population[i], -5, 5)
                f_i = func(x_i)
                f_values.append(f_i)
                calls += 1

                if f_i < best_fx:
                    best_fx = f_i
                    best_x = x_i.copy()

            # If we ran out of time during evaluation, break
            if len(f_values) < lambda_:
                break

            # Sort by fitness
            sorted_indices = np.argsort(f_values)
            y_sorted = population[sorted_indices]
            f_sorted = np.array(f_values)[sorted_indices]

            # Update CMA-ES
            update_CMAES(y_sorted, f_sorted)

        return best_x, best_fx
# EVOLVE-BLOCK-END
