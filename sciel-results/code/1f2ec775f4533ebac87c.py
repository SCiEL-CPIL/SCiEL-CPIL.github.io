"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, budget, dim):
        rng = np.random.default_rng(42)
        
        # Parameters
        lambda_ = max(4, int(4 + 3 * np.log(dim)))  # Population size
        mu = lambda_ // 2  # Number of selected individuals
        c_sigma = (mu_eff + 2) / ((dim + 1.3) ** 2 + mu_eff)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        mu_eff = sum((1.0 / np.arange(1, mu + 1)) ** 2)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        c_cov_pos = (2 + 2 * mu_eff / dim) / ((dim + 1.5) ** 2 + mu_eff)
        c_cov_neg = 2 * (1 - c_cov_pos) * min(1, (mu_eff - 2 + 1 / mu_eff) / (dim ** 2 + dim))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Initial state
        x = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        best_x = x.copy()
        best_f = func(x)
        
        calls = 1
        for gen in range(budget - 1):
            if calls >= budget:
                break
            
            # Generate offspring
            ys = []
            fs = []
            for i in range(lambda_):
                if calls >= budget:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * z @ np.linalg.cholesky(C).T
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                ys.append(y)
                fs.append(fy)
            
            if not ys:
                break
            
            ys = np.array(ys)
            fs = np.array(fs)
            
            # Update best
            idx_best = np.argmin(fs)
            if fs[idx_best] < best_f:
                best_f = fs[idx_best]
                best_x = ys[idx_best].copy()
            
            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = ys[order]
            fs_sorted = fs[order]
            
            # Update mean
            y_w = np.sum(weights[:, np.newaxis] * ys_sorted[:mu], axis=0)
            x = x + y_w
            
            # Update step size control
            y_sigma = (y_w / sigma) @ np.linalg.cholesky(C).T
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            
            # Update covariance matrix
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) / chi_n) < (1.4 + 2 / (dim + 1))
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma
            rank_one = np.outer(pc, pc)
            rank_mu = np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(ys_sorted[:mu] - x, ys_sorted[:mu] - x), axis=0) / (sigma ** 2)
            C = (1 + c_cov_pos * c_c * (1 - c_c) - c_cov_neg) * C + c_cov_pos * (rank_one - C) + c_cov_neg * rank_mu
            
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10)
            
            # Clip mean to box
            x = np.clip(x, -5, 5)
        
        return best_x, best_f

    def __call__(self, func):
        return self._cma_es(func, self.budget, self.dim)
# EVOLVE-BLOCK-END
