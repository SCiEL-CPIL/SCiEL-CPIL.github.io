"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES for black-box continuous optimization on BBOB-style f12."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        # Parameters
        # For n=5, lambda is typically 4+floor(3*log(n)) = 4+3=7, but let's use a standard value
        # lambda = 4 + floor(3 * log(n))
        lam = max(4 + int(3 * np.log(n)), 10)
        
        # We need to be careful with the budget.
        # Each generation costs 'lam' evaluations.
        # We have 'budget' total evaluations.
        # Initial point costs 1 evaluation.
        # Remaining budget: budget - 1
        # Number of generations: (budget - 1) // lam
        
        rng = np.random.default_rng(12345)
        
        # Initialization
        # Start near the center, as the minimizer is in [-4, 4]^5
        x_mean = np.zeros(n)
        # Initial sigma: scale to cover the box roughly
        sigma = 1.0
        
        # Covariance matrix
        C = np.eye(n)
        # Square root of C (initially identity)
        Cs = np.eye(n)
        
        # Weights for recombination
        # Standard CMA-ES weights: w_i proportional to log(lam/2 + 1) - log(i) for i=1..lam/2
        # Here we use a simpler, robust weighting scheme often used in implementations
        # mu = floor(lam / 2)
        mu = lam // 2
        weights = np.zeros(lam)
        # Compute weights for the top mu individuals
        # Standard formula: w_i = log(mu + 1) - log(i) for i=1..mu
        for i in range(mu):
            weights[i] = np.log(mu + 1) - np.log(i + 1)
        weights /= np.sum(weights)
        
        # mu_eff = 1 / sum(w_i^2)
        mu_eff = 1.0 / np.sum(weights[:mu]**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Damping for cumulative step
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        
        # Threshold for h_sigma
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        # State variables
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Initial evaluation
        x0 = np.clip(x_mean, -5, 5)
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0
        
        remaining_budget = budget - 1
        gen = 0
        
        while remaining_budget >= lam:
            # Sample lambda points
            Y = np.zeros((lam, n))
            X = np.zeros((lam, n))
            F = np.zeros(lam)
            
            for i in range(lam):
                # Sample from N(0, C)
                # Use Cholesky or SVD of Cs. Since Cs is maintained, we use it directly.
                # y = Cs @ z
                z = rng.standard_normal(n)
                y = Cs @ z
                x_new = x_mean + sigma * y
                # Clip to box
                x_new = np.clip(x_new, -5, 5)
                X[i] = x_new
                F[i] = func(x_new)
                remaining_budget -= 1
                
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = x_new.copy()
            
            # Sort by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            # Compute new mean
            # Only use top mu individuals
            x_mean_new = np.sum(weights[:mu, np.newaxis] * X_sorted[:mu], axis=0)
            
            # Compute evolution path
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (1/sigma) * Cs.T @ (x_mean_new - x_mean)
            # Note: The term (x_mean_new - x_mean) is in the original coordinates.
            # The step in the standardized space is (x_mean_new - x_mean) / sigma
            # But we need to transform it back to the isotropic space using Cs.T
            # Let's follow the standard CMA-ES update:
            # p_sigma = (1-c_sigma)*p_sigma + sqrt(c_sigma*(2-c_sigma)*mu_eff) * Cs.T @ (x_mean_new - x_mean) / sigma
            
            diff = (x_mean_new - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (Cs.T @ diff)
            
            # Check for h_sigma (heavy-tailed distribution check)
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-c_sigma)^{2*gen+2}) < (1.4 + 2/(n+1)) * chi_n
            # Approximation for large gen: sqrt(1 - (1-c_sigma)^{2*gen}) approaches 1
            # For simplicity and stability, we often just check the norm against chi_n
            norm_ps = np.linalg.norm(p_sigma)
            # Threshold
            thresh = (1.4 + 2 / (n + 1)) * chi_n
            h_sigma = 1.0 if norm_ps / np.sqrt(1 - (1 - c_sigma)**(2 * gen + 2)) < thresh else 0.0
            
            # Update p_c
            # p_c = (1 - c_c) * p_c + h_sigma * sqrt(c_c * (2 - c_c) * mu_eff) * Cs.T @ (x_mean_new - x_mean) / sigma
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (Cs.T @ diff)
            
            # Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c * p_c.T + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * sum w_i * y_i * y_i.T
            # Where y_i = (x_i - x_mean_old) / sigma
            # We need to compute the weighted sum of outer products of the standardized steps
            
            # Compute Y (standardized steps)
            # Y[i] = (X[i] - x_mean_old) / sigma
            # But we only use the top mu
            Y_std = np.zeros((mu, n))
            for i in range(mu):
                Y_std[i] = (X_sorted[i] - x_mean) / sigma
            
            # Update C
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C)
            
            # Add the weighted sum of outer products
            # sum_{i=1}^mu w_i * y_i * y_i.T
            # We can do this as (Y_std.T * weights[:mu]) @ Y_std
            C += c_mu * (Y_std.T @ (weights[:mu, np.newaxis] * Y_std))
            
            # Ensure C is symmetric and positive definite (numerical stability)
            C = (C + C.T) / 2.0
            
            # Update Cs (square root of C)
            # Use eigen decomposition or Cholesky
            # Since C is SPD, Cholesky is faster, but eigen is more robust to small negative eigenvalues
            # Let's try Cholesky first, fallback to eigen
            try:
                L = np.linalg.cholesky(C)
                Cs = L
            except np.linalg.LinAlgError:
                # Fallback: eigen decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip small negative eigenvalues to 0
                eigvals = np.clip(eigvals, 1e-16, None)
                Cs = eigvecs @ np.diag(np.sqrt(eigvals))
            
            # Update sigma
            # sigma = sigma * exp(c_sigma / d_sigma * (||p_sigma|| / chi_n - 1))
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            # Clip sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update mean
            x_mean = x_mean_new
            
            # Check convergence
            # If sigma is very small or the range of fitness is very small, we can stop
            # But we must use the full budget if possible to maximize AOCC?
            # Actually, AOCC rewards early progress. If we converge, we can just keep evaluating the best point?
            # No, evaluating the same point doesn't help. But we must make exactly budget calls?
            # "More than 500 calls... invalid". "Repeating the last n_i if fewer calls are made."
            # So we can stop early if we are stuck, but we should try to use the budget wisely.
            # If sigma is too small, the search is dead.
            if sigma < 1e-6:
                # We can break and let the remaining budget be "wasted" (or we can just return)
                # But the loop condition checks remaining_budget >= lam.
                # If we break, we exit the loop.
                break
            
            gen += 1
        
        # If we have remaining budget but can't do a full generation, we can do single-step (1+1) ES
        while remaining_budget > 0:
            # Simple random walk
            y = np.clip(x_mean + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)
            remaining_budget -= 1
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
            # Update mean slightly towards best?
            # For simplicity, just keep x_mean as is or move towards best
            # x_mean = x_mean + 0.1 * (best_x - x_mean)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
