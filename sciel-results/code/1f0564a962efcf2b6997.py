"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase optimizer: coarse random search + CMA-ES refinement for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(12345)

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        # Phase 1: Coarse random search to identify basin of attraction
        # Use ~30% of budget for exploration
        phase1_budget = max(50, int(budget * 0.3))
        
        # Start with origin
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls += 1
        best_f = f0
        best_x = x0.copy()
        
        # Random search with adaptive step size
        sigma = 3.0
        for i in range(phase1_budget - 1):
            if time.time() > deadline:
                break
            if calls >= budget:
                break
            
            # Generate random point around current best
            y = best_x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
            
            # Adapt sigma: decrease if we're making progress, increase if stuck
            # Simple adaptation: reduce sigma gradually
            sigma *= 0.98
            sigma = max(sigma, 0.5)

        # Phase 2: CMA-ES refinement
        remaining = budget - calls
        if remaining < 10:
            return best_x, best_f
        
        # Initialize CMA-ES parameters
        lambda_ = min(10, remaining // 2)
        mu = max(1, lambda_ // 2)
        w = np.zeros(mu)
        w[:mu] = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1.0 / np.sum(w**2)
        
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 - mueff))
        
        x_mean = best_x.copy()
        sigma = 0.5  # Initial step size for refinement
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        p_c = np.zeros(dim)
        p_sigma = np.zeros(dim)
        hsig = (1 + 1.4 / dim + 2 / ((dim + 1) ** 2)) * np.sqrt((dim + 1) / (dim - 1))
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            pop = np.zeros((lambda_, dim))
            fvals = np.zeros(lambda_)
            
            for i in range(lambda_):
                if time.time() > deadline:
                    break
                if calls >= budget:
                    break
                
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                pop[i] = y
                fvals[i] = func(y)
                calls += 1
            
            if calls >= budget:
                break
            
            # Sort by fitness
            order = np.argsort(fvals)
            pop_sorted = pop[order]
            fvals_sorted = fvals[order]
            
            # Update best
            if fvals_sorted[0] < best_f:
                best_f = fvals_sorted[0]
                best_x = pop_sorted[0].copy()
            
            # Compute weighted mean
            x_new = np.sum(w[:, np.newaxis] * pop_sorted[:mu], axis=0)
            
            # Update covariance matrix
            y_vec = (x_new - x_mean) / sigma
            C = (1 - c1 - cmu) * C + c1 * (np.outer(p_c, p_c) - (1 - cc) * C) + \
                cmu * np.sum(w[:, np.newaxis] * np.outer(pop_sorted[:mu] - x_mean, pop_sorted[:mu] - x_mean) / sigma**2, axis=0)
            C = (C + C.T) / 2
            
            # Update path variables
            p_c = (1 - cc) * p_c + cc * y_vec
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * np.linalg.norm(y_vec)
            
            # Update sigma
            sigma *= np.exp(cs / dsigma * (np.linalg.norm(p_sigma) / hsig - 1))
            dsigma = 1 + 2 * max(0, (mueff - 1) / (dim + 3))
            
            # Update x_mean
            x_mean = x_new
            x_mean = np.clip(x_mean, -5, 5)
            
            # Recompute C_sqrt periodically
            if calls % 20 == 0:
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-10, None)
                    C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                except:
                    C_sqrt = np.eye(dim)
                    C = np.eye(dim)
            
            # Keep sigma in bounds
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # If sigma is too small, restart
            if sigma < 1e-6:
                x_mean = best_x + 0.3 * rng.standard_normal(dim)
                x_mean = np.clip(x_mean, -5, 5)
                sigma = 0.5
                C = np.eye(dim)
                C_sqrt = np.eye(dim)
                p_c = np.zeros(dim)
                p_sigma = np.zeros(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
