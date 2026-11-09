"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with PCA-initialized covariance for ill-conditioned landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin

        # --- Phase 1: Initial sampling to estimate landscape structure ---
        # Sample a small set of points to estimate the principal axes
        n_init = 20
        init_points = []
        init_vals = []
        for _ in range(n_init):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            init_points.append(x)
            init_vals.append(fx)
        
        init_points = np.array(init_points)
        init_vals = np.array(init_vals)
        
        # Compute covariance of points, weighted by inverse fitness (lower fitness = higher weight)
        # Use a softmax-like weighting to focus on better points
        min_val = np.min(init_vals)
        weights = np.exp(-0.1 * (init_vals - min_val))
        weights /= np.sum(weights)
        
        # Weighted mean
        mean_init = np.average(init_points, axis=0, weights=weights)
        
        # Weighted covariance
        centered = init_points - mean_init
        cov_init = np.zeros((dim, dim))
        for i in range(n_init):
            cov_init += weights[i] * np.outer(centered[i], centered[i])
        cov_init /= (np.sum(weights) - 1) if np.sum(weights) > 1 else 1.0
        
        # Ensure positive semi-definite
        eigvals, eigvecs = np.linalg.eigh(cov_init)
        eigvals = np.maximum(eigvals, 1e-8)
        cov_init = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        # --- Phase 2: CMA-ES ---
        # Initial point: best from initial sampling
        best_idx = np.argmin(init_vals)
        x = init_points[best_idx].copy()
        fx = init_vals[best_idx]
        
        # Initial sigma based on spread of initial points
        sigma = np.mean(np.linalg.norm(init_points - mean_init, axis=1)) * 0.5
        sigma = np.clip(sigma, 0.1, 2.0)
        
        # CMA-ES parameters
        lambda_ = 8  # population size
        mu = lambda_ // 2  # number of parents
        # Weights for recombination
        weights = np.ones(mu)
        weights = np.arange(mu, 0, -1)
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Update rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial covariance matrix: blend identity with PCA-estimated covariance
        # Scale PCA covariance to have same trace as identity * sigma^2
        trace_pca = np.trace(cov_init)
        trace_target = dim * sigma**2
        if trace_pca > 1e-8:
            C = (trace_target / trace_pca) * cov_init
        else:
            C = sigma**2 * np.eye(dim)
        
        # Ensure C is symmetric and PSD
        C = (C + C.T) / 2
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-10)
        C = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        # Cholesky factor
        try:
            C_sqrt = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C_sqrt = np.sqrt(sigma**2) * np.eye(dim)
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        calls_made = n_init  # count initial calls
        best_x = x.copy()
        best_fx = fx
        
        # Main loop
        while calls_made < budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Generate population
            offspring = []
            offspring_vals = []
            
            for i in range(lambda_):
                # Check deadline mid-generation
                if time.time() > deadline:
                    break
                
                z = rng.standard_normal(dim)
                y = x + sigma * C_sqrt @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                
                offspring.append(y)
                offspring_vals.append(fy)
                
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
            
            if len(offspring) < mu:
                break
            
            # Sort offspring by fitness
            offspring = np.array(offspring)
            offspring_vals = np.array(offspring_vals)
            order = np.argsort(offspring_vals)
            offspring = offspring[order[:mu]]
            offspring_vals = offspring_vals[order[:mu]]
            
            # Weighted mean
            x_new = np.average(offspring, axis=0, weights=weights)
            
            # Update sigma and covariance
            C_sqrt_old = C_sqrt
            
            # p_sigma update
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_old @ (x_new - x) / sigma)
            
            # p_c update
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma
            
            # Rank-one update
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made - n_init + 1))) < (1 + 3 / dim) * np.sqrt((dim + 2) / (1 - (1 - c_sigma)**2)) else 0.0
            
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            
            # Rank-mu update
            y_sorted = (offspring - x) / sigma
            C_update = c_1 * (np.outer(p_c, p_c) - C) + delta_h * C
            C_update += c_mu * np.sum(weights[:, None, None] * np.einsum('i,ij,ik->jk', y_sorted, y_sorted, y_sorted), axis=0)
            
            # Ensure PSD
            eigvals, eigvecs = np.linalg.eigh(C_update)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update sigma
            sigma *= np.exp((c_sigma / 5) * (np.linalg.norm(C_sqrt_old @ (x_new - x)) / sigma - np.sqrt(dim)))
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Update C_sqrt
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C_sqrt = np.sqrt(sigma**2) * np.eye(dim)
            
            # Update mean
            x = x_new
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
