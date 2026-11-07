"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer for BBOB f19 (Griewank-Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # Initial population size
        lam = max(4, int(4 + 3 * np.log(n)))
        
        # CMA-ES parameters
        mu = lam // 2  # number of selected individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        # Initial point and step size
        rng = np.random.default_rng(42)
        x = np.zeros(n)
        sigma = 2.0
        ps = np.zeros(n)
        pc = np.zeros(n)
        
        # Covariance matrix
        C = np.eye(n)
        C_sqrt = np.eye(n)
        C_inv = np.eye(n)
        
        # Precompute
        def sqrtm(mat):
            # Simple eigenvalue decomposition for symmetric positive definite
            vals, vecs = np.linalg.eigh(mat)
            vals = np.maximum(vals, 1e-12)
            return vecs @ np.diag(np.sqrt(vals)) @ vecs.T

        def invm(mat):
            try:
                return np.linalg.inv(mat)
            except:
                return np.eye(n)
        
        # First call
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        
        for gen in range(budget):
            # Generate population
            Y = np.zeros((lam, n))
            F = np.zeros(lam)
            for i in range(lam):
                # Sample from N(0, C)
                z = rng.standard_normal(n)
                y = x + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                Y[i] = y
                F[i] = func(y)
                
                if F[i] < best_fx:
                    best_fx = F[i]
                    best_x = y.copy()
            
            # Sort
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]
            
            # Update mean
            x_new = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)
            
            # Check if we've used up budget
            if gen == budget - 1:
                break
            
            # Update evolution paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_sqrt.T @ (x_new - x))
            hsig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * chi_n else 0
            
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (x_new - x)
            
            # Update covariance
            rank_one = np.outer(x_new - x, x_new - x)
            rank_mu = np.sum(weights[:, None, None] * np.outer(Y_sorted[:mu] - x, Y_sorted[:mu] - x), axis=0)
            
            C = (1 + c_1 * hsig * (hsig - 1) + c_mu) * C + c_1 * (np.outer(pc, pc) - C) + c_mu * (rank_mu - C)
            
            # Ensure positive definiteness
            try:
                C_sqrt = sqrtm(C)
                C_inv = invm(C)
            except:
                C_sqrt = np.eye(n)
                C_inv = np.eye(n)
            
            x = x_new
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Early stopping check
            if sigma < 1e-12:
                break
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
