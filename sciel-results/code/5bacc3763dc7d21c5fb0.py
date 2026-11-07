"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for BBOB f12."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280  # 300s limit, 20s safety margin
        
        # CMA-ES Parameters
        dim = self.dim
        # Population size: 4 + floor(3 * ln(n))
        lam = int(4 + 3 * np.log(dim))
        # Initial step size: sqrt(5) is often good for [-5,5] box
        sigma = 2.0 
        # Initial mean: center of the box
        mean = np.zeros(dim)
        
        # Covariance matrix: identity
        C = np.eye(dim)
        # Inverse square root of C (for sampling)
        Cinv_sqrt = np.eye(dim)
        
        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Step sizes for paths
        # mu is the number of recombination weights
        mu = lam // 2
        # Weights: log(1) - log(mu)
        weights = np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        # mu_eff
        mueff = 1.0 / np.sum(weights**2)
        
        # Adaptation rates
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        # Chi_n: expected value of norm of standard normal
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim**2))
        
        # Initial evaluation
        x0 = np.clip(mean, -5, 5)
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0
        calls_made = 1
        
        # Pre-compute sampling matrix update logic
        # We need to update Cinv_sqrt periodically
        
        while calls_made < self.budget:
            if time.time() > deadline:
                break
                
            # 1. Sample lambda points
            # Update sampling matrix (Cinv_sqrt) if needed or every iteration for simplicity (small dim)
            # For dim=5, computing eigendecomposition is cheap
            eigvals, eigvecs = np.linalg.eigh(C)
            # Avoid zero eigenvalues
            eigvals = np.maximum(eigvals, 1e-12)
            Cinv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            Y = rng.standard_normal((lam, dim))
            X = mean + sigma * (Y @ Cinv_sqrt.T)
            
            # Clip to box
            X = np.clip(X, -5, 5)
            
            # 2. Evaluate
            F = np.array([func(xi) for xi in X])
            calls_made += lam
            
            # Update best
            idx_best = np.argmin(F)
            if F[idx_best] < best_f:
                best_f = F[idx_best]
                best_x = X[idx_best].copy()
                
            # Check termination criteria
            # If sigma is very small or progress is negligible
            if sigma < 1e-10:
                break
            
            # 3. Update mean (weighted recombination)
            order = np.argsort(F)[:mu]
            mean_old = mean.copy()
            mean = np.sum(weights[:, np.newaxis] * X[order], axis=0)
            
            # 4. Update sigma (CSA - Covariance Matrix Self-Adaptation is implicit in C update)
            # Actually standard CMA-ES updates sigma using ps
            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (Cinv_sqrt @ (mean - mean_old))
            
            # Hessian norm correction
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made + 1) / lam)) < (1.4 + 2 / (dim + 1)) * chi_n)
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # 5. Update pc
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) / sigma * (mean - mean_old)
            
            # 6. Update C
            # Rank-one update
            y_mean = (mean - mean_old) / sigma
            C += c_1 * (np.outer(y_mean, y_mean) + (1 - h_sigma) * c_c * (2 - c_c) * C - np.eye(dim))
            
            # Rank-mu update
            Y_rec = np.array([Cinv_sqrt @ (xi - mean_old) / sigma for xi in X[order]])
            C += c_mu * np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(Y_rec, Y_rec), axis=0)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Add small identity to prevent singularity
            C += 1e-12 * np.eye(dim)
            
            # If sigma becomes too large, clamp
            if sigma > 10.0:
                sigma = 10.0
            if sigma < 1e-6:
                sigma = 1e-6

        return best_x, best_f
# EVOLVE-BLOCK-END
