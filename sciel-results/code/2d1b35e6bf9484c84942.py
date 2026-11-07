"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES implementation for BBOB-style optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))
        
        # Initial center
        # BBOB instances have hidden shifts. Starting at 0 is often a good heuristic 
        # for "Attractive sector" (f6) which usually has a well-defined basin, 
        # but random start is safer against shifts. 
        # Let's start at 0,0,0,0,0 as f6 is typically centered or shifted modestly.
        # Actually, the prompt says "hidden seeded shift... minimizer lies in [-4, 4]^5".
        # Starting at 0 is a reasonable guess.
        x = np.zeros(dim)
        
        # Initial sigma
        # The box is [-5, 5]. A sigma of 2.0 covers a good chunk of the space.
        sigma = 2.0
        
        # Weights
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1 / np.sum(weights**2)
        
        # Learning rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        # State variables
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        dnorm = np.sqrt(dim) * (1 + 1/(2*dim) - 1/(4*dim**2))
        
        best_x = x.copy()
        best_f = func(x)
        
        calls_made = 1
        
        # Main loop
        while calls_made < budget:
            if time.time() > deadline - 5.0:
                break
                
            # Generate population
            # Eigen decomposition for sampling
            # For dim=5, this is cheap.
            eigvals, eigvecs = np.linalg.eigh(C)
            # Handle numerical issues
            eigvals = np.clip(eigvals, 1e-12, None)
            
            B = eigvecs
            D = np.sqrt(eigvals)
            
            # Sample
            Y = np.zeros((lam, dim))
            for i in range(lam):
                zi = np.random.randn(dim)
                Y[i] = B @ (D * zi)
                
            X = x + sigma * Y
            F = np.array([func(xi) for xi in X])
            calls_made += lam
            
            # Sort
            idx = np.argsort(F)
            X = X[idx]
            F = F[idx]
            
            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = X[0].copy()
            
            # Update mean
            xnew = np.sum(weights[:, None] * X[:mu], axis=0)
            
            # Update path evolution strategies
            pc = (1 - cc) * pc + cc * np.sqrt(mueff) / (sigma) * (xnew - x)
            cs_norm = np.linalg.norm(pc) / dnorm
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (xnew - x)
            
            # Update covariance matrix
            # Rank-1 update
            C = (1 + c1 * cs * (cs_norm**2 - 1) * cc - c1) * C
            C += c1 * np.outer(pc, pc)
            
            # Rank-mu update
            v = (xnew - x) / sigma
            C += cmu * np.sum(weights[:mu] * np.outer(v[:mu], v[:mu]), axis=0)
            
            # Symmetrize to remove numerical noise
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(cs / dnorm * (np.linalg.norm(ps) / dnorm - 1))
            
            # Update mean
            x = xnew
            
            # Convergence check
            if np.all(np.abs(X[:mu] - X[mu:]) < 1e-10) or sigma < 1e-12:
                break
                
        return best_x, best_f
# EVOLVE-BLOCK-END
