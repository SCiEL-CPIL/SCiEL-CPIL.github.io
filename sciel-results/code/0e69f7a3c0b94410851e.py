"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES implementation for BBOB-style Sphere f1."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Time guard
        t0 = time.time()
        deadline = t0 + 290.0

        rng = np.random.default_rng(42)

        # CMA-ES Parameters
        # Population size lambda
        lam = 4 + int(3 * np.log(dim))
        # Selection size mu
        mu = lam // 2
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w**2)
        # Step size control
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff + 1) / (dim + 1)) - 1) + c_sigma
        # Mean control
        c1 = (2 / ((dim + 1.3)**2 + mueff)) * (1 - ccm)
        cmu = min(1 - c1, (2 * (mueff - 2 + 1/mueff)) / ((dim + 2)**2 + mueff))
        ccm = (mueff + 2) / (dim + mueff + 5)
        cs = (mueff + 1.7) / (dim + mueff + 2.5)
        # Damping for mean update
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Initial mean: center of the box
        x_mean = np.zeros(dim)
        sigma = 2.0  # Initial step size, box is [-5, 5]^5, so 2 is reasonable
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        # Covariance matrix
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        
        calls_made = 0
        global_best_x = x_mean.copy()
        global_best_f = func(x_mean)
        calls_made += 1
        
        # Weights for mean update
        w_mean = np.zeros(mu)
        w_mean[0] = 1.0
        w_mean[1:] = 0.0 # Standard CMA-ES uses only the best or weighted top mu
        # Actually, standard CMA-ES uses weighted average of top mu
        w_mean = w.copy()
        
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            X = np.zeros((lam, dim))
            F = np.zeros(lam)
            
            for i in range(lam):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                z = rng.standard_normal(dim)
                x = x_mean + sigma * (C_sqrt @ z)
                # Clip to box
                x = np.clip(x, -5, 5)
                X[i] = x
                F[i] = func(x)
                calls_made += 1
                
                if F[i] < global_best_f:
                    global_best_f = F[i]
                    global_best_x = x.copy()
            
            if calls_made >= budget:
                break
                
            # Sort by fitness
            idx = np.argsort(F)
            F_sorted = F[idx]
            X_sorted = X[idx]
            
            # Update mean
            x_new = np.sum(w_mean[:, np.newaxis] * X_sorted[:mu], axis=0)
            
            # Update sigma
            z_mean = np.linalg.solve(C_sqrt, (x_new - x_mean) / sigma)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * z_mean
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update covariance matrix
            p_c = (1 - ccm) * p_c + np.sqrt(ccm * (2 - ccm) * mueff) * np.linalg.solve(C_sqrt, (x_new - x_mean) / sigma)
            
            h_sig = 1 if np.linalg.norm(p_c) / np.sqrt(1 - (1 - ccm)**((calls_made // lam + 1) * ccm)) < (1 + 2/dim) * chi_n else 0
            
            C = (1 + c1 * h_sig + cmu) * C - c1 * np.outer(p_c, p_c)
            for i in range(mu):
                y_i = np.linalg.solve(C_sqrt, (X_sorted[i] - x_mean) / sigma)
                C -= cmu * w[i] * np.outer(y_i, y_i)
            
            # Ensure positive definiteness
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update square root and inverse square root
            C_sqrt, C_inv_sqrt = self._update_chol(C, C_sqrt, C_inv_sqrt)
            
            x_mean = x_new
        
        return global_best_x, global_best_f

    def _update_chol(self, C, C_sqrt, C_inv_sqrt):
        # Simple update using eigen decomposition for stability
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.sqrt(eigvals)
        C_sqrt = eigvecs @ np.diag(eigvals)
        C_inv_sqrt = eigvecs @ np.diag(1/eigvals)
        return C_sqrt, C_inv_sqrt
# EVOLVE-BLOCK-END
