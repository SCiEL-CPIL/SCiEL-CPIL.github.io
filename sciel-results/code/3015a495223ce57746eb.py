"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (Success-Weighted Covariance Matrix Adaptation)"""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget

    def __call__(self, func):
        start_time = time.time()
        time_limit = 325.0  # Safety margin before 330s kill
        
        rng = np.random.default_rng(0)
        
        # Initialize center and covariance
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        C = np.eye(self.dim)
        
        # CMA-ES parameters
        # Standard CMA-ES uses lambda = 4 + floor(3 * log(10)) ~ 14
        lam = 14
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        
        # Step size
        sigma = 2.0
        
        # Time constants
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2)**2 + mu_eff))
        
        # Cumulation parameters
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        
        # Precompute inverse square root of C (approximate or exact)
        # For simplicity and speed, we use Cholesky or eigendecomposition if needed, 
        # but standard CMA-ES updates C directly. 
        # We need to sample from N(0, C). We can use Cholesky of C.
        
        calls_made = 1
        
        while calls_made < self.budget:
            if time.time() - start_time > time_limit:
                break
            
            # Generate population
            X = np.zeros((lam, self.dim))
            F = np.zeros(lam)
            
            # Get Cholesky factor of C for sampling
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback if C is not positive definite (should be rare with damping)
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                L = eigvecs @ np.diag(np.sqrt(eigvals))
            
            for i in range(lam):
                z = rng.standard_normal(self.dim)
                X[i] = x + sigma * (L @ z)
                X[i] = np.clip(X[i], -5, 5)
                F[i] = func(X[i])
                calls_made += 1
                
                # If we hit budget mid-population, break
                if calls_made >= self.budget:
                    lam = i + 1
                    break
            
            # Sort by fitness
            idx = np.argsort(F)
            X = X[idx]
            F = F[idx]
            weights = weights[:lam]
            
            # Update mean
            w_sum = np.sum(weights)
            x_new = np.sum(weights[:, np.newaxis] * X[:mu], axis=0) / w_sum
            
            # Update step size (sigma) using cs adaptation
            # Need C^{-1/2} x
            try:
                C_inv_sqrt_x = np.linalg.solve(np.linalg.cholesky(C), x_new - x)
            except:
                C_inv_sqrt_x = x_new - x
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt_x
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made / lam))) 
                    < (1 + 2 / (self.dim + 1)) * np.sqrt(self.dim + 1 - 1))
            
            if hsig:
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma
            else:
                pc = (1 - c_c) * pc
                
            # Update Covariance
            delta_sig = (np.linalg.norm(ps) ** 2 / self.dim - 1) * c_sigma * (2 - c_sigma)
            C = (1 + c_1 + c_mu - c_sigma * delta_sig) * C
            
            rank_mu = np.sum(weights[:mu]**2, axis=0) # This is wrong, weights are 1D
            # rank_mu is scalar sum of squares of top mu weights
            rank_mu_val = np.sum(weights[:mu]**2)
            
            rank_one = pc[:, np.newaxis] * pc.T
            rank_mu_mat = np.sum(weights[:mu][:, np.newaxis] * (X[:mu] - x)[:, np.newaxis, :] * (X[:mu] - x)[:, :, np.newaxis], axis=0) / sigma**2
            
            C += c_1 * (rank_one - (1 - c_c) * C) + c_mu * (rank_mu_mat - C)
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            # Add small identity for stability if needed
            C += 1e-12 * np.eye(self.dim)
            
            # Update state
            x, fx = x_new, F[0]
            sigma *= np.exp(c_sigma / (5 * self.dim) * (np.linalg.norm(ps) / np.sqrt(self.dim) - 1))
            sigma = np.clip(sigma, 1e-10, 10)
            
        return x, fx
# EVOLVE-BLOCK-END
