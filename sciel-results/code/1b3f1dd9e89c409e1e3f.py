"""CMA-ES based optimizer for f19 with time-aware budget management."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for the Griewank-Rosenbrock function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 285  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initial population: mix of random and structured points
        n_init = 10
        X_init = np.random.uniform(-5, 5, (n_init, dim))
        # Add some structured points (Halton-like or grid)
        for i in range(dim):
            X_init[i, i] = 0.0
        Y_init = np.array([func(x) for x in X_init])
        calls = n_init

        best_x = X_init[np.argmin(Y_init)]
        best_y = Y_init.min()

        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * log(dim))
        lambda_pop = 4 + int(3 * np.log(dim))
        lambda_pop = max(lambda_pop, 10)  # At least 10

        # Initial step size: based on initial spread
        sigma = 1.0
        x_mean = np.zeros(dim)
        
        # Initialize covariance matrix as identity
        C = np.eye(dim)
        # Inverse of C
        C_inv = np.eye(dim)
        
        # Step sizes for C and C_inv
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 - mu_eff + c_sigma) / (dim + 4)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Initial mean: use the best initial point
        x_mean = best_x.copy()
        
        # Main CMA-ES loop
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Check if we have enough time for one iteration
            # Each iteration evaluates lambda_pop points
            if calls + lambda_pop > budget:
                # Do a final small search near the best point
                n_remaining = budget - calls
                for _ in range(n_remaining):
                    if time.time() > deadline:
                        break
                    x_new = best_x + rng.normal(0, 0.1, dim)
                    x_new = np.clip(x_new, -5, 5)
                    y_new = func(x_new)
                    calls += 1
                    if y_new < best_y:
                        best_y = y_new
                        best_x = x_new.copy()
                break
            
            # Generate population
            # Update step size using sigma model
            # For simplicity, we'll use a fixed sigma that adapts based on progress
            
            # Generate lambda_pop points
            X_pop = np.zeros((lambda_pop, dim))
            Y_pop = np.zeros(lambda_pop)
            
            for i in range(lambda_pop):
                # Sample from multivariate normal with covariance C
                # Use Cholesky decomposition
                try:
                    L = np.linalg.cholesky(C)
                    z = rng.standard_normal(dim)
                    x_i = x_mean + sigma * (L @ z)
                except np.linalg.LinAlgError:
                    x_i = x_mean + sigma * rng.standard_normal(dim)
                
                x_i = np.clip(x_i, -5, 5)
                X_pop[i] = x_i
                Y_pop[i] = func(x_i)
                calls += 1
                
                if Y_pop[i] < best_y:
                    best_y = Y_pop[i]
                    best_x = x_i.copy()
            
            # Sort by fitness
            order = np.argsort(Y_pop)
            X_sorted = X_pop[order]
            Y_sorted = Y_pop[order]
            
            # Update mean
            x_mean_new = np.sum(weights[:, np.newaxis] * X_sorted[:mu], axis=0)
            
            # Update evolution paths
            # p_sigma update
            c_sigma_sqrt = np.sqrt(c_sigma * (2 - c_sigma) * mu_eff)
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma_sqrt * (np.linalg.inv(np.sqrt(C)) @ (x_mean_new - x_mean)) / sigma
            
            # Adapt sigma
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # p_c update
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (np.linalg.inv(np.sqrt(C)) @ (x_mean_new - x_mean)) / sigma
            
            # Update covariance matrix
            # Rank-1 update
            y_mean = (x_mean_new - x_mean) / sigma
            delta_sig = (np.linalg.norm(p_sigma) / chi_n - 1) * c_sigma
            C = (1 + c_1 * delta_sig) * C + c_1 * np.outer(y_mean, y_mean)
            
            # Rank-mu update
            z_matrix = np.zeros((mu, dim))
            for i in range(mu):
                z_matrix[i] = (X_sorted[i] - x_mean) / sigma
            C = C + c_mu * np.sum(weights[:mu, np.newaxis, np.newaxis] * z_matrix[:, np.newaxis, :] * z_matrix[:, np.newaxis, :], axis=0)
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update mean
            x_mean = x_mean_new
            
            # Ensure C is positive definite by adding small regularization
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-8:
                C += (1e-8 - np.min(eigvals)) * np.eye(dim)
            
            # Update C_inv
            try:
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                C_inv = np.eye(dim)
            
            # Clip sigma to reasonable range
            sigma = np.clip(sigma, 0.01, 5.0)
            
            # Clip mean to bounds
            x_mean = np.clip(x_mean, -5, 5)

        return best_x, best_y
# EVOLVE-BLOCK-END
