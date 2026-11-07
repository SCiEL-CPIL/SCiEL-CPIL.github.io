"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for the Bent Cigar function."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        rng = np.random.default_rng(123)
        dim = self.dim
        
        # Initial mean: center of the box
        mean = np.zeros(dim)
        
        # Initial step size (sigma)
        # Heuristic for CMA-ES: sigma0 = 0.3 * (max - min)
        sigma = 3.0 
        
        # Population size
        # Standard CMA-ES: lambda = 4 + floor(3 * log(n))
        lam = 4 + int(3 * np.log(dim))
        
        # Weights
        mu = lam // 2
        weights = np.log(np.arange(mu, 0, -1)) - np.log(mu)
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / (dim**1.7 + 10) + c_sigma
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / (dim**1.8 + 10))
        
        # State variables
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # Initial function evaluation
        f_mean = func(mean)
        best_x = mean.copy()
        best_f = f_mean
        
        calls_made = 1
        
        # Loop until budget or time limit
        while calls_made < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Sample population
            # Compute inverse square root of C for sampling
            # Use eigendecomposition for stability
            evals, evecs = np.linalg.eigh(C)
            # Clip small negative eigenvalues due to numerical errors
            evals = np.maximum(evals, 1e-12)
            C_inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
            
            # Generate z_i
            Z = rng.standard_normal((lam, dim))
            # Transform to Y_i
            Y = Z @ C_inv_sqrt.T
            # X_i = mean + sigma * Y_i
            X = mean[:, np.newaxis] + sigma * Y.T
            # Clip to box
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            calls_made += lam
            
            # Sort
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = X_sorted[0].copy()
            
            # Update mean
            old_mean = mean.copy()
            mean_new = np.sum(weights[:, np.newaxis] * X_sorted[:mu], axis=0)
            
            # Update sigma
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_inv_sqrt * (mean_new - old_mean))
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_inv_sqrt @ (mean_new - old_mean))
            if np.linalg.norm(ps) / chi_n < (1.4 + 2 / (dim + 1)) * (1 - c_sigma):
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                
            # Update C
            # pc = (1 - c_c) * pc + sqrt(c_c * (2 - c_c) * mu_eff) * (mean_new - old_mean) / sigma
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean_new - old_mean) / sigma
            
            # Rank-1 update
            C += c_1 * (np.outer(pc, pc) - C + (1 - (1 - c_c)**2) * np.eye(dim))
            
            # Rank-mu update
            # diff = X_sorted[:mu] - old_mean
            diff = X_sorted[:mu] - old_mean
            C += c_mu * np.sum(weights[:, np.newaxis, np.newaxis] * diff[:, np.newaxis, :] * diff[:, np.newaxis, :].T, axis=0)
            
            # Symmetrize C
            C = (C + C.T) / 2
            
            mean = mean_new
            f_mean = np.sum(weights * F_sorted[:mu])
            
            # If sigma becomes too small, stop
            if sigma < 1e-12:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
