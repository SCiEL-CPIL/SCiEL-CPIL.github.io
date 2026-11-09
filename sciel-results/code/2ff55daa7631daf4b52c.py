"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with PCA-initialized covariance for Schaffers F7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit

        # CMA-ES parameters
        lambda_ = 4  # population size
        mu = lambda_ // 2  # number of parents
        # Weights for recombination
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        # Step size adaptation parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        # Covariance adaptation parameters
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        # Chi-squared correction
        chi_lambda = np.sqrt(lambda_) * (1 - 1 / (4 * lambda_) + 1 / (21 * lambda_**2))

        # Initial point: sample a few points to estimate the landscape anisotropy
        rng = np.random.default_rng(42)
        
        # Initial sigma and center
        sigma = 1.0
        x_mean = np.zeros(dim)
        
        # PCA initialization: sample a small set of points to estimate principal axes
        # Use a ring of points around the center to capture anisotropy
        n_pca = 20
        pca_points = []
        for i in range(n_pca):
            direction = rng.standard_normal(dim)
            direction /= np.linalg.norm(direction)
            point = x_mean + 3.0 * direction
            point = np.clip(point, -5, 5)
            pca_points.append(point)
        
        # Evaluate PCA points
        pca_values = []
        for p in pca_points:
            if time.time() > deadline:
                break
            pca_values.append(func(p))
        
        # Find best PCA point as initial center
        if len(pca_values) > 0:
            best_idx = np.argmin(pca_values)
            x_mean = pca_points[best_idx].copy()
            f_best = pca_values[best_idx]
        else:
            x_mean = np.zeros(dim)
            f_best = func(x_mean)

        # Estimate covariance from PCA points weighted by fitness
        # Use the top half of points to estimate the direction of the funnel
        if len(pca_values) >= 4:
            # Sort by value
            sorted_idx = np.argsort(pca_values)
            # Use the best half
            n_best = max(2, len(pca_values) // 2)
            best_points = np.array([pca_points[i] for i in sorted_idx[:n_best]])
            
            # Compute covariance of these points
            if len(best_points) >= dim:
                cov_est = np.cov(best_points.T)
                if cov_est.shape == (dim, dim) and np.all(np.isfinite(cov_est)):
                    # Regularize
                    cov_est += 1e-6 * np.eye(dim)
                    # Cholesky decomposition
                    try:
                        C_chol = np.linalg.cholesky(cov_est)
                        # Normalize: we want the covariance to be approximately identity * sigma^2
                        # but aligned with the principal axes
                        # Extract the direction matrix
                        # C_chol * C_chol.T = cov_est
                        # We want B such that B @ B.T = cov_est
                        # So B = C_chol
                        # But we need to scale it appropriately
                        # For CMA-ES, C = B @ B.T where C is the covariance matrix
                        # Initially C should be close to I, but aligned
                        # Let's normalize the directions
                        norms = np.linalg.norm(C_chol, axis=0)
                        if np.all(norms > 1e-10):
                            C_chol_normed = C_chol / norms
                            # Use this as the initial transformation
                            B = C_chol_normed
                        else:
                            B = np.eye(dim)
                    except:
                        B = np.eye(dim)
                else:
                    B = np.eye(dim)
            else:
                B = np.eye(dim)
        else:
            B = np.eye(dim)

        # CMA-ES state
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Number of evaluations used
        n_evals = len(pca_values)
        
        # Main CMA-ES loop
        while n_evals < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            pop = []
            pop_values = []
            
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (B @ z)
                y = np.clip(y, -5, 5)
                pop.append(y)
            
            # Evaluate population
            for y in pop:
                if n_evals >= budget or time.time() > deadline:
                    break
                fy = func(y)
                pop_values.append(fy)
                n_evals += 1
            
            if len(pop_values) < lambda_:
                break
            
            # Sort by fitness
            sorted_idx = np.argsort(pop_values)
            sorted_pop = [pop[i] for i in sorted_idx]
            sorted_vals = [pop_values[i] for i in sorted_idx]
            
            # Update best
            f_best = sorted_vals[0]
            x_best = sorted_pop[0]
            
            # Recombination
            x_new = sum(w[i] * sorted_pop[i] for i in range(mu))
            
            # Update sigma (CSA)
            h_sigma = (np.sqrt(1 - (1 - c_sigma)**2 * (budget - n_evals)) / 
                       (1 - c_sigma**2 * (budget - n_evals)))
            # Actually, standard CSA:
            # h_sigma = 1 if norm(p_sigma) / sqrt(1 - (1-c_sigma)^2 * t) < (1.4 + 2/(dim+1)) * chi_lambda
            # Simplified:
            p_sigma_old = p_sigma.copy()
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ (x_new - x_mean) / sigma)
            norm_p_sigma = np.linalg.norm(p_sigma)
            # Expected norm
            expected_norm = chi_lambda
            # h_sigma condition
            h_sigma = 1.0 if norm_p_sigma / np.sqrt(1 - (1 - c_sigma)**(2 * (n_evals))) < (1.4 + 2.0 / (dim + 1)) * expected_norm else 0.0
            
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_lambda - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 100.0)
            
            # Update C (CSA for covariance)
            if h_sigma:
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x_mean) / sigma
            
            # Rank-1 and rank-mu update
            delta_h = 1 - h_sigma
            C = B @ B.T
            C = C + c_1 * (1 + (1 - h_sigma) * c_c * (2 - c_c)) * np.outer(x_mean - x_mean, x_mean - x_mean)  # This is 0
            # Actually:
            C = B @ B.T
            # Rank-1
            C = (1 + c_1 * (1 - h_sigma) * c_c * (2 - c_c)) * C + c_1 * np.outer((x_new - x_mean) / sigma, (x_new - x_mean) / sigma)
            # Wait, the standard update is:
            # C = (1 + c_1*(1-h_sigma)*c_c*(2-c_c)) * C + c_1 * (x_new - x_mean) @ (x_new - x_mean).T
            # But x_new - x_mean is in the transformed space? No, in the original space.
            # Actually, in CMA-ES, the update is:
            # C = (1 - c_1 - c_mu) * C + c_1 * (x_new - x_mean) @ (x_new - x_mean).T + c_mu * sum w_i (y_i - x_mean) @ (y_i - x_mean).T
            # Let me redo this properly.
            
            # Reset C to B @ B.T
            C = B @ B.T
            
            # Rank-1 update
            diff = (x_new - x_mean)
            C = C + c_1 * (1 + (1 - h_sigma) * c_c * (2 - c_c)) * np.outer(diff, diff)
            
            # Rank-mu update
            for i in range(mu):
                diff_i = (sorted_pop[i] - x_mean)
                C = C + c_mu * w[i] * np.outer(diff_i, diff_i)
            
            # Decay
            C = (1 - c_1 - c_mu) * C + c_1 * (1 + (1 - h_sigma) * c_c * (2 - c_c)) * np.outer(diff, diff) + c_mu * sum(w[i] * np.outer(sorted_pop[i] - x_mean, sorted_pop[i] - x_mean) for i in range(mu))
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Cholesky decomposition
            try:
                B = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            except:
                # Regularize and retry
                C = C + 1e-6 * np.eye(dim)
                try:
                    B = np.linalg.cholesky(C)
                except:
                    B = np.eye(dim)
            
            # Update mean
            x_mean = x_new
            
            # Check convergence
            if sigma < 1e-10:
                break

        return x_best if 'x_best' in locals() else x_mean, f_best if 'f_best' in locals() else float('inf')
# EVOLVE-BLOCK-END
