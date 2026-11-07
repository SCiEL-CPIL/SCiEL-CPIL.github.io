"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer with restarts for multimodal functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        # CMA-ES parameters
        pop_size = 9 + int(3 * np.log(n))
        
        # Adaptation parameters
        cc = (4 + n / 3) / (n + 4 + 2 * n / (n + 1))
        cmu = min(1 - 2 / (n + 2), 2 / ((n + 2) ** 2 + 1))
        cs = (n + 1.3) / (n + 3 + 2 * n / (n + 1))
        ds = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1) + cs
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        # Weights
        w = np.log(pop_size / 2 + 0.5) - np.log(np.arange(1, pop_size + 1))
        w /= np.sum(w)
        mueff = int(np.sum(w[:pop_size // 2]))
        
        max_restarts = 5
        restarts = 0
        
        while calls_used < budget and restarts < max_restarts:
            # Initialize state for this restart
            if restarts == 0:
                m = rng.uniform(-5, 5, n)
                sigma = 2.0
            else:
                m = best_x.copy()
                sigma = 0.5
            
            C = np.eye(n)
            ps = np.zeros(n)
            pc = np.zeros((n, n))
            
            # Precompute weights
            w_sum = np.sum(w)
            
            while calls_used < budget:
                # Get Cholesky factor B such that C = B B^T
                try:
                    B = np.linalg.cholesky(C)
                    B_inv = np.linalg.inv(B)
                except np.linalg.LinAlgError:
                    B = np.eye(n)
                    B_inv = np.eye(n)
                
                # Generate population
                Y = np.zeros((pop_size, n))
                for i in range(pop_size):
                    z = rng.standard_normal(n)
                    Y[i] = m + sigma * (B @ z)
                    Y[i] = np.clip(Y[i], -5, 5)
                
                # Evaluate
                F = np.zeros(pop_size)
                for i in range(pop_size):
                    if calls_used >= budget:
                        break
                    F[i] = func(Y[i])
                    calls_used += 1
                    if F[i] < best_f:
                        best_f = F[i]
                        best_x = Y[i].copy()
                
                if calls_used >= budget:
                    break
                
                # Sort
                order = np.argsort(F)
                Y_sorted = Y[order]
                F_sorted = F[order]
                
                # Update mean
                m_old = m.copy()
                m = np.sum(w[:, np.newaxis] * Y_sorted, axis=0)
                
                # Update step size
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B_inv @ (m - m_old) / sigma)
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_used // pop_size)))) < (1.4 + 2 / (n + 1)) * chi_n
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
                
                # Update covariance
                pc = (1 - cmu) * pc + hsig * cmu * (2 - cmu) * np.outer(m - m_old, m - m_old) / (sigma ** 2)
                C = (1 + cmu * (2 - cmu) - cs) * C + cs * (pc - C) + cmu * (np.sum(w[:mueff]) * np.sum(w[:mueff, np.newaxis] * np.outer(Y_sorted[:mueff] - m_old, Y_sorted[:mueff] - m_old), axis=0) / (sigma ** 2))
                
                # Symmetrize and regularize
                C = (C + C.T) / 2
                C += 1e-9 * np.eye(n)
                
                # Check stagnation
                if F_sorted[0] >= best_f - 1e-10:
                    # No improvement in this generation
                    if not hasattr(self, '_stagnation'):
                        self._stagnation = 0
                    self._stagnation += 1
                else:
                    self._stagnation = 0
                
                if self._stagnation > 5 or sigma < 1e-6:
                    restarts += 1
                    break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
