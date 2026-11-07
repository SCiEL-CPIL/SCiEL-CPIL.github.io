"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES based optimizer for BBOB f3 (Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))
        # Initial mean: Center of the box is a reasonable guess, though shift is hidden.
        # The problem states minimizer is in [-4, 4]^5. 0 is a good prior.
        mean = np.zeros(dim)
        # Initial sigma: Domain is [-5, 5], width 10. Sigma ~ 2.5 covers most of it.
        sigma = 2.5
        # Covariance matrix: Identity
        C = np.eye(dim)
        # Inverse square root of C
        C_inv_sqrt = np.eye(dim)
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Learning rates
        mu = lam // 2  # Number of recombination weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size adaptation
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Covariance matrix adaptation
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_cov = (2 + 2 * mu_eff / dim) / (dim**1.8 + mu_eff + 13)
        
        # Counter for generations
        gen = 0
        calls_made = 0
        best_x = mean.copy()
        best_f = np.inf
        
        # Time check function
        def time_left():
            return deadline - time.time()
        
        # Evaluate a single point
        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            calls_made += 1
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Main Loop
        while calls_made + lam <= budget and time_left() > 1.0:
            # Generate population
            # We need to generate lam samples from N(mean, sigma^2 * C)
            # To do this efficiently, we transform standard normals
            # y = z @ C_inv_sqrt.T
            # x = mean + sigma * y
            
            # Generate standard normal samples
            Z = np.random.randn(lam, dim)
            
            # Transform to correlated samples
            Y = Z @ C_inv_sqrt.T
            
            # Create population
            X = mean + sigma * Y
            
            # Evaluate population
            F = np.zeros(lam)
            for i in range(lam):
                # Clip to box [-5, 5]
                x_i = np.clip(X[i], -5, 5)
                f_i = evaluate(x_i)
                if f_i is None:
                    # Budget exhausted mid-generation
                    break
                F[i] = f_i
            
            if calls_made >= budget:
                break
                
            # Sort indices by fitness
            idx = np.argsort(F)
            X = X[idx]
            F = F[idx]
            
            # Update mean using recombination weights
            old_mean = mean.copy()
            mean = np.sum(weights[:, np.newaxis] * X[:mu], axis=0)
            
            # Update evolution path for sigma
            # p_sigma = (1 - c_sigma) * p_sigma + c_sigma * sqrt(mu_eff) * (C_inv_sqrt @ (mean - old_mean) / sigma)
            delta_mean = (mean - old_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * (C_inv_sqrt @ delta_mean)
            
            # Update sigma
            # chi_n = expected norm of standard normal in n-dim
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update evolution path for covariance
            # p_c = (1 - c_c) * p_c + c_c * sqrt(mu_eff) * (C_inv_sqrt @ (mean - old_mean) / sigma)
            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * (C_inv_sqrt @ delta_mean)
            
            # Update covariance matrix
            # h_sigma = (||p_sigma|| / sqrt(1 - (1-c_sigma)^{2*(gen+1)})) / chi_n < 1.7 + 2/(dim+1)
            # Approximation for h_sigma
            h_sigma = (np.linalg.norm(p_sigma) / chi_n) < (1.7 + 2.0 / (dim + 1))
            
            # Rank-one update
            rank_one = np.outer(p_c, p_c)
            if h_sigma:
                rank_one += (1 / c_c - 1 - c_c) * np.outer(p_sigma, p_sigma)
            
            # Rank-mu update
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_i = C_inv_sqrt @ (X[i] - old_mean) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)
            
            C = (1 + c_cov * (mu_eff - 1) - c_cov) * C + c_cov * (rank_one + rank_mu)
            
            # Ensure C is symmetric and positive definite (numerical stability)
            C = (C + C.T) / 2.0
            
            # Update C_inv_sqrt
            # We need to compute the inverse square root of C
            # Using eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip eigenvalues to prevent negative values due to numerical errors
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            gen += 1
            
            # Optional: If sigma becomes too small, we might be stuck.
            # But CMA-ES usually handles this. If we detect stagnation, we could restart,
            # but with 500 budget, a single run is preferred to avoid wasting calls.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
