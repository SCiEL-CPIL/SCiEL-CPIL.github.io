"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for Rastrigin-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights**2)
        
        sigma = 1.0
        x_mean = np.zeros(dim)
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        best_x = x_mean.copy()
        best_f = float('inf')
        
        calls = 0
        
        x_pop = np.zeros((lam, dim))
        f_pop = np.zeros(lam)
        
        while calls < budget:
            for i in range(lam):
                if calls >= budget:
                    break
                z = rng.standard_normal(dim)
                x_pop[i] = x_mean + sigma * (B @ (D * z))
                x_pop[i] = np.clip(x_pop[i], -5, 5)
                
                f_pop[i] = func(x_pop[i])
                calls += 1
            
            if calls >= budget:
                break
                
            order = np.argsort(f_pop)
            x_pop = x_pop[order]
            f_pop = f_pop[order]
            
            if f_pop[0] < best_f:
                best_f = f_pop[0]
                best_x = x_pop[0].copy()
                
            x_new_mean = np.zeros(dim)
            for i in range(mu):
                x_new_mean += weights[i] * x_pop[i]
            x_new_mean = np.clip(x_new_mean, -5, 5)
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (B @ (D @ (B.T @ x_new_mean) / sigma))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls + 1) / (dim + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1)))
            if hsig:
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * (B.T @ x_new_mean / sigma)
            
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
            
            delta_diag = 1 if hsig else 0
            rank_one = np.outer(pc, pc)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_i = B.T @ x_pop[i] / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)
            
            C = (1 + c_1 * delta_diag + c_mu - c_1) * C + c_1 * rank_one + c_mu * rank_mu
            C = (C + C.T) / 2
            C += 1e-9 * np.eye(dim)
            
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                D = np.sqrt(eigvals)
                B = eigvecs
            except np.linalg.LinAlgError:
                D = np.ones(dim)
                B = np.eye(dim)
                C = np.eye(dim)
                
            x_mean = x_new_mean
            
        return best_x, best_f
# EVOLVE-BLOCK-END
