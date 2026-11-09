"""CMA-ES optimizer for Bent Cigar (f12) with adaptive covariance matrix."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES optimizer adapted for the anisotropic landscape of the Bent Cigar function.
    Uses a full covariance matrix update to efficiently navigate the elongated valley.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))  # ~9 for dim=5
        # Weights for the best individuals
        mu = lam // 2
        weights = np.log(np.arange(mu, 0, -1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Update rates
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2 + mueff)**2 - 2))
        cc = (4 + mueff/dim) / (dim + 4 + mueff)
        
        # Initial state
        rng = np.random.default_rng(42)
        x_mean = rng.uniform(-3, 3, dim)
        sigma = 2.0
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        
        # Evolve paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = x_mean.copy()
        best_fx = func(x_mean)
        calls_used = 1
        
        # Pre-compute Cholesky factor of C
        try:
            C_sqrt = np.linalg.cholesky(C)
        except:
            C_sqrt = np.eye(dim)
            
        while calls_used < budget:
            if time.time() >= deadline:
                break
                
            # Generate population
            ys = []
            fs = []
            
            for i in range(lam):
                if calls_used >= budget or time.time() >= deadline:
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1
                ys.append(y)
                fs.append(fy)
                
            if not ys:
                break
                
            # Sort by fitness
            idx = np.argsort(fs)
            ys = [ys[i] for i in idx]
            fs = [fs[i] for i in idx]
            
            # Update best
            if fs[0] < best_fx:
                best_fx = fs[0]
                best_x = ys[0].copy()
                
            # Update mean
            x_mean_old = x_mean
            x_mean = sum([w * y for w, y in zip(weights, ys[:mu])])
            
            # Update sigma
            z_mean = (x_mean - x_mean_old) / sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * (C_inv @ z_mean)
            sigma *= np.exp(cs / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            # d_sigma is usually 1.0 + 2/max(0, dim/n - 1) + mueff/(2*dim)
            d_sigma = 1.0 + 2.0 / max(0, dim / mu - 1) + mueff / (2.0 * dim)
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            
            # Update covariance matrix
            y_norm = np.linalg.norm(z_mean)
            h_sigma = 1 if y_norm < (1 + 0.5*cs) * chi_n else 0
            
            p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mueff) * (C_inv @ z_mean)
            
            # Rank-one and rank-mu update
            C = (1 - cs*(c1 + cmu) - cmu) * C \
                + cs * np.outer(p_c, p_c) \
                + cmu * sum([w * np.outer(y - x_mean_old, y - x_mean_old) / sigma**2 for w, y in zip(weights, ys[:mu])])
            
            # Ensure positive definiteness
            try:
                C_inv = np.linalg.inv(C)
                C_sqrt = np.linalg.cholesky(C)
            except:
                # If singular, reset C to identity and increase sigma slightly
                C = np.eye(dim)
                C_inv = np.eye(dim)
                C_sqrt = np.eye(dim)
                sigma *= 1.5

        return best_x, best_fx
# EVOLVE-BLOCK-END
