"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import warnings


class Optimizer:
    """CMA-ES style optimizer using scipy's differential evolution as a robust fallback,
    but primarily using a custom CMA-ES implementation for better convergence on BBOB f12."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        
        # Initialize CMA-ES parameters
        dim = self.dim
        # Start from center of the box
        x_mean = np.zeros(dim)
        
        # Initial sigma based on box size
        sigma = 2.0
        
        # Population size
        lam = max(4 + int(3 * np.log(dim)), 6)
        mu = lam // 2
        
        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        
        # Variance reduction factor
        mueff = 1 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Covariance matrix
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        best_x = x_mean.copy()
        best_f = np.inf
        
        # Track number of function evaluations
        nfe = 0
        
        # Evaluate initial point
        f_val = func(x_mean)
        nfe += 1
        best_f = f_val
        best_x = x_mean.copy()
        
        # CMA-ES main loop
        while nfe < self.budget:
            # Generate population
            B = rng.standard_normal((lam, dim))
            # Apply eigenvalue decomposition of C (approximate for simplicity)
            # For efficiency, we use a simplified update
            pops = x_mean + sigma * B
            
            # Clip to box
            pops = np.clip(pops, -5, 5)
            
            # Evaluate population
            fvals = np.array([func(p) for p in pops])
            nfe += lam
            
            # Sort by fitness
            order = np.argsort(fvals)
            y_sorted = pops[order]
            f_sorted = fvals[order]
            
            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = y_sorted[0].copy()
            
            # Recombination
            y_w = np.sum(weights[:, np.newaxis] * y_sorted[:mu], axis=0)
            
            # Update mean
            x_mean = y_w
            
            # Update step size control vector
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / chi * np.dot(np.linalg.cholesky(C).T, (y_w - x_mean) / sigma)
            
            # Update covariance matrix
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * (y_w - x_mean) / sigma
            
            delta_heaviside = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (self.budget - nfe + 1))) < (1.4 + 2 / (dim + 1)) * chi else 0
            
            C = ((1 - c_1 - c_mu) * C + 
                 c_1 * (C + pc @ pc.T - C) + 
                 c_mu * np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(y_sorted[:mu] - x_mean, y_sorted[:mu] - x_mean), axis=0) / sigma ** 2 +
                 c_sigma * (1 + (1 - c_sigma) * d_sigma * delta_heaviside) * np.outer(ps, ps))
            
            # Symmetrize C
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi - 1))
            
            # Ensure sigma is positive and reasonable
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Check for convergence
            if sigma < 1e-8:
                break
            
            # If we're close to the boundary, adjust
            if np.all(np.abs(x_mean) > 4.5):
                # Try a random restart
                x_mean = rng.uniform(-5, 5, dim)
                sigma = 2.0
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
