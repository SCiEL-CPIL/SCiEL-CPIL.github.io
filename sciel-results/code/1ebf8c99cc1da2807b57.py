"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # For dim=5, lambda=12 is a good balance
        lam = 12
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size control parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Covariance matrix adaptation parameters
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial state
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        
        sigma = 2.0
        C = np.eye(dim)
        Csqrt = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Precompute constants
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Main loop
        for gen in range(budget // lam + 2):
            if time.time() > deadline:
                break
                
            # Sample population
            population = []
            for i in range(lam):
                z = np.random.standard_normal(dim)
                y = x + sigma * (Csqrt @ z)
                y = np.clip(y, -5, 5)
                population.append(y)
            
            # Evaluate
            fvals = np.array([func(y) for y in population])
            
            # Track global best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_fx:
                best_fx = fvals[min_idx]
                best_x = population[min_idx].copy()
            
            # Sort
            order = np.argsort(fvals)
            population = [population[i] for i in order]
            fvals = fvals[order]
            
            # Weighted mean
            m = sum(weights[i] * population[i] for i in range(mu))
            
            # Update step size
            dnorm = np.linalg.norm(Csqrt @ m)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / dnorm * (Csqrt @ m)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2*(gen+1))) < (1.4 + 2/(dim+1)) * chi_n)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * m
            
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C + c_1 * np.outer(pc, pc)
            
            for i in range(mu):
                yi = (population[i] - x) / sigma
                C += c_mu * weights[i] * np.outer(yi, yi)
            
            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.any(eigvals < 0):
                C = eigvecs @ np.diag(np.maximum(eigvals, 1e-10)) @ eigvecs.T
            
            # Update Csqrt (SVD approximation)
            U, s, Vt = np.linalg.svd(C)
            Csqrt = U @ np.diag(np.sqrt(s)) @ Vt
            
            # Update mean
            x = m
            
            # Check termination criteria (optional, but good for efficiency)
            if sigma < 1e-12:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
