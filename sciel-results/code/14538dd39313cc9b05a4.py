"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES for BBOB f1 (Sphere)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        d = self.dim
        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(d))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        c1 = (2.0 / ((d + 1.3)**2 + mueff))
        cmu = min(1.0 - c1, (2.0 * (mueff - 2.0 + 1.0/mueff)) / ((d + 2.0)**2 + mueff))
        cs = (mueff + 2.0) / (d + mueff + 5.0)
        ds = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (d + 1.0)) - 1.0) + cs
        cc = (4.0 + mueff / d) / (d + 4.0 + 2.0 * mueff / d)
        c_sigma = (mueff + 2.0) / (d + mueff + 5.0)
        
        # Initial state
        m = np.zeros(d)
        sigma = 2.0  # Initial step size, box is [-5, 5]
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        chiN = np.sqrt(d) * (1.0 - 1.0 / (8.0 * d) + 1.0 / (129.0 * d**2))
        
        # Initial population
        # We will generate one population per iteration.
        # Budget is 500. Lambda is 8. So we can do ~60 generations.
        
        best_x = None
        best_f = np.inf
        
        # We need to manage the budget carefully.
        # Each generation costs lambda_ evaluations.
        # Total budget: self.budget
        
        while self.budget > 0 and time.time() < deadline:
            # Sample lambda points
            # We need to generate lambda points from N(m, sigma^2 C)
            # Use Cholesky decomposition for C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset
                C = np.eye(d)
                L = np.eye(d)
                ps = np.zeros(d)
                pc = np.zeros(d)
                
            # Generate standard normal
            z = np.random.standard_normal((lambda_, d))
            # Transform to search points
            y = z @ L.T
            x = m + sigma * y.T
            x = x.T  # Shape (lambda_, d)
            
            # Evaluate
            f_vals = np.array([func(xi) for xi in x])
            
            # Update best
            idx_min = np.argmin(f_vals)
            if f_vals[idx_min] < best_f:
                best_f = f_vals[idx_min]
                best_x = x[idx_min].copy()
                
            self.budget -= lambda_
            
            if self.budget < 0:
                break
                
            # Sort
            sort_idx = np.argsort(f_vals)
            y_sorted = y[sort_idx]
            f_sorted = f_vals[sort_idx]
            
            # Update mean
            m_new = m + sigma * np.sum(weights[:, np.newaxis] * y_sorted[:mu], axis=0)
            
            # Update step size (CSA)
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) / sigma * (L.T @ (m_new - m))
            if np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs)**2) < (1.4 + 2.0 / (d + 1.0)) * chiN:
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chiN - 1.0))
                
            # Update covariance (CMA)
            hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs)**2) < (1.4 + 2.0 / (d + 1.0)) * chiN)
            hsig = float(hsig)
            
            pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * (m_new - m) / sigma
            
            # Rank-mu update
            y_mu = y_sorted[:mu]
            C = (1.0 + c1 * hsig * (hsig - 1.0) + cmu) * C
            C += c1 * np.outer(pc, pc)
            C += cmu * np.sum(weights[:, np.newaxis]**2 * np.outer(y_mu, y_mu), axis=0)
            
            # Symmetrize
            C = (C + C.T) / 2.0
            
            m = m_new
            
            # Safety check for C
            if not np.all(np.isfinite(C)):
                break
                
        return best_x, best_f
# EVOLVE-BLOCK-END
