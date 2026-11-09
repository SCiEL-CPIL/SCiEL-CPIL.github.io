"""Rewritten optimizer for BBOB f6 (Attractive sector) using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Continuous optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls_made = 0
        x_best = None
        f_best = np.inf

        def eval_func(x):
            nonlocal calls_made, x_best, f_best
            if calls_made >= budget:
                return np.inf
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            if f < f_best:
                f_best = f
                x_best = x.copy()
            return f

        # CMA-ES Parameters
        # Standard population size lambda = 4 + floor(3 * log(n))
        lam = 4 + int(3 * np.log(dim))
        # Initial step size
        sigma = 2.0  # Start relatively wide to cover [-5, 5]
        # Mean vector initialization
        mean = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Adaptation parameters (standard defaults)
        mu = lam // 2  # Number of weighted parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Step size adaptation (CSA) parameters
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Covariance matrix adaptation parameters
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        # Evolution path parameters
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # History vectors
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Generation counter
        generation = 0
        
        # Run CMA-ES
        while calls_made < budget:
            # Generate offspring
            offspring = []
            fitnesses = []
            
            # We can potentially generate more than lam if we have budget left, 
            # but standard CMA-ES uses fixed lam. 
            # To use budget efficiently, we just run generations until budget is exhausted.
            
            for i in range(lam):
                if calls_made >= budget:
                    break
                # Sample from multivariate normal
                z = rng.standard_normal(dim)
                # Apply Cholesky decomposition of C
                # We need to update Cholesky factor every generation
                # For efficiency in this short script, we compute it once per generation
                # Actually, we should compute Cholesky of C once before the inner loop
                # But C changes. Let's compute L = cholesky(C) here.
                # Optimization: Compute L once per generation.
                
                # Wait, C is constant during the sampling of the generation.
                # So we should compute L before the for loop.
                pass
            
            # Let's restructure: Compute L once per generation
            # But we need to handle the budget constraint carefully.
            
            # Recompute Cholesky factor
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, add small regularization
                C = C + 1e-6 * np.eye(dim)
                L = np.linalg.cholesky(C)
                
            # Sample offspring
            offspring = []
            for i in range(lam):
                if calls_made >= budget:
                    break
                z = rng.standard_normal(dim)
                x_new = mean + sigma * (L @ z)
                x_new = np.clip(x_new, -5, 5)
                offspring.append(x_new)
                
            if len(offspring) == 0:
                break
                
            # Evaluate offspring
            fitnesses = []
            for x in offspring:
                f = eval_func(x)
                fitnesses.append(f)
                
            if calls_made >= budget:
                break
                
            # Sort by fitness
            indices = np.argsort(fitnesses)
            sorted_offspring = [offspring[i] for i in indices]
            sorted_fitnesses = [fitnesses[i] for i in indices]
            
            # Select best mu
            selected = sorted_offspring[:mu]
            selected_fitnesses = sorted_fitnesses[:mu]
            
            # Update mean
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * np.array(selected), axis=0)
            
            # Update evolution path for sigma
            # p_sigma = (1 - c_sigma) * p_sigma + c_sigma * sqrt(mueff) * C^(-1/2) * (mean - old_mean) / sigma
            # We need C^(-1/2). We can use L to compute this efficiently.
            # C^(-1/2) = L^(-T) @ L^(-1) ? No.
            # Let's just compute the transformation.
            # y = (mean - old_mean) / sigma
            y = (mean - old_mean) / sigma
            # Transform y with C^(-1/2)
            # This is equivalent to solving L^T L x = y => x = L^(-T) L^(-1) y ?
            # Actually, if C = L L^T, then C^(-1/2) is not simply L^(-T).
            # We can compute C_inv_sqrt using eigendecomposition or by solving linear systems.
            # For small dim, we can just compute the inverse square root.
            # Or simpler: update p_sigma using the standard formula involving the inverse of C.
            # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mueff) * C^(-1/2) (mean - old_mean)/sigma
            
            # Let's compute C_inv_sqrt
            # Using eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip eigenvalues to avoid division by zero
            eigvals = np.clip(eigvals, 1e-10, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ y)
            
            # Check sigma adaptation
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (generation + 1))) < (1.4 + 2 / (dim + 1)) * chi_n:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                
            # Update evolution path for covariance
            # p_c = (1 - c_c) p_c + sqrt(c_c (2 - c_c) mueff) * (mean - old_mean) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * y
            
            # Update covariance matrix
            # delta_sigma = (1 - c_c) / c_c * (||p_c||^2 / dim - 1)
            delta_sigma = (1 - c_c) / c_c * (np.dot(p_c, p_c) / dim - 1)
            
            # Rank-one update
            C = (1 + c_1 * delta_sigma - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c)
            
            # Rank-mu update
            # C += c_mu * sum(weights_i * (y_i * y_i^T) - C_old) ?
            # Formula: C = (1 + c_1 * delta_sigma - c_1 - c_mu) C + c_1 p_c p_c^T + c_mu sum w_i (x_i - mean_old)(x_i - mean_old)^T
            # Note: The term sum w_i (x_i - mean_old)(x_i - mean_old)^T is computed from the selected parents.
            # Let's compute the sum of outer products of the selected parents relative to old_mean.
            # y_selected = (selected - old_mean) / sigma
            y_selected = (np.array(selected) - old_mean[None, :]) / sigma
            H = np.sum(weights[:, None, None] * y_selected[:, :, None] * y_selected[:, None, :], axis=0)
            
            C = C + c_mu * H
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            # Regularize if needed
            eigvals, _ = np.linalg.eigh(C)
            if np.min(eigvals) < 1e-10:
                C = C + 1e-10 * np.eye(dim)
                
            generation += 1
            
            # Optional: Restart logic if stagnation is detected
            # For simplicity and given the budget, we rely on the standard CMA-ES stopping criteria or just running out of budget.
            # However, if the step size becomes too small or too large, we might restart.
            # Let's add a simple restart if sigma is very small or very large for a few generations.
            # But for now, let's stick to a single run to keep it simple and robust.

        return x_best, f_best
# EVOLVE-BLOCK-END
