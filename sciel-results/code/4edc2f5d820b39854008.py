"""Rewritten optimizer using CMA-ES for BBOB-style f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for BBOB-style f4 (Büche-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES Parameters
        lam = 8
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = (weights ** 2).sum() ** -1
        
        cs = (lam / 2.0 + 1.0) / (dim + lam / 2.0 + 5.0)
        cc = (4.0 + lam / dim) / (dim + 4.0 + 2.0 * lam / dim)
        c1 = 2.0 / ((dim + 1.3) ** 2 + lam)
        cmu = min(1.0 - c1, (2.0 * (mueff - 2.0 + 1.0 / mueff)) / ((dim + 2.0) ** 2 + mueff))
        
        # Initial State
        x = np.zeros(dim)
        sigma = 1.0
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        evals_used = 0
        best_x = x.copy()
        best_f = float('inf')
        
        while evals_used + lam <= budget:
            # Generate population
            pop = np.zeros((lam, dim))
            fitness = np.zeros(lam)
            y_vectors = np.zeros((lam, dim))
            
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = B.dot(D * z)
                y_vectors[i] = y
                xi = x + sigma * y
                xi_clipped = np.clip(xi, -5.0, 5.0)
                pop[i] = xi_clipped
                fitness[i] = func(xi_clipped)
                evals_used += 1
                
                if fitness[i] < best_f:
                    best_f = fitness[i]
                    best_x = xi_clipped.copy()
            
            # Sort population
            idx = np.argsort(fitness)
            pop = pop[idx]
            fitness = fitness[idx]
            y_top = y_vectors[idx][:mu]
            
            # Update mean
            x_new = np.sum(weights[:, np.newaxis] * pop[:mu], axis=0)
            
            # Update step size (CSA)
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) / sigma * (x_new - x)
            
            # Update pc
            gen = evals_used / lam
            factor = 1.0 - (1.0 - cs) ** (2.0 * (gen + 1.0))
            thresh = (1.4 + 2.0 / (dim + 1.0)) * np.sqrt(factor)
            norm_ps = np.linalg.norm(ps)
            hsig = 1.0 if (norm_ps / np.sqrt(factor) < thresh) else 0.0
            
            if hsig:
                pc = (1.0 - cc) * pc + cc * (x_new - x) / sigma
            
            # Update Covariance
            y_w = y_top * np.sqrt(weights)[:, np.newaxis]
            outer_sum = y_w.T.dot(y_w)
            
            C_new = (1.0 - c1 - cmu) * C + c1 * (np.outer(ps, ps) + (hsig - 1.0) * np.outer(pc, pc)) + cmu * outer_sum
            C_new = (C_new + C_new.T) / 2.0
            
            # Eigen-decomposition for next iteration
            D2, B = np.linalg.eigh(C_new)
            D2 = np.clip(D2, 1e-16, None)
            D = np.sqrt(D2)
            
            # Update state
            x = x_new
            C = C_new
            B = B
            D = D
            
        return best_x, best_f
# EVOLVE-BLOCK-END
