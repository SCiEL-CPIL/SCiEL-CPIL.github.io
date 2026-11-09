"""
CMA-ES implementation for minimizing a black-box function on [-5, 5]^dim.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 325.0  # 5s safety margin

        # CMA-ES Parameters
        # Standard lambda for dim=5 is around 11
        lam = max(4, int(4 + 3 * np.log(dim)))
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Initialization
        # Start at the center of the box, which is a good guess for shifted Rastrigin
        m = np.zeros(dim)
        sigma = 2.0  # Initial step size
        
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Precompute inverse of C_sqrt for sampling
        # We will update C and its decomposition each generation
        
        best_x = m.copy()
        best_f = func(m)
        
        evals = 1
        start_time = time.time()
        
        while evals < budget and time.time() < deadline:
            # Sample points
            # z ~ N(0, I)
            Z = np.random.standard_normal((lam, dim))
            # y = m + sigma * C_sqrt @ z
            Y = m + sigma * (Z @ C_sqrt.T)
            
            # Clip to bounds
            Y = np.clip(Y, -5, 5)
            
            # Evaluate
            F = np.array([func(y) for y in Y])
            evals += lam
            
            # Sort by fitness
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]
            
            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = Y[0].copy()
            
            # Update mean
            w = weights
            m_old = m.copy()
            m = np.sum(Y[:mu] * w[:, np.newaxis], axis=0)
            
            # Update step size control variable
            z = (m - m_old) / (sigma * np.linalg.norm(C_sqrt))
            # Note: strictly speaking, z should be in the standard normal space
            # A common approximation is using the norm of the step in the transformed space
            # Let's use the standard update rule
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / ||C_sqrt|| * (m - m_old)/sigma
            # We need ||C_sqrt|| = sqrt(trace(C))
            norm_C_sqrt = np.sqrt(np.trace(C))
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / norm_C_sqrt * (m - m_old) / sigma
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (evals + 1) / (dim + c_sigma)))) < (1 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1))
            
            # Update evolution path for covariance
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) / norm_C_sqrt * (m - m_old) / sigma
            
            # Update covariance matrix
            # Delta_h = (hsig - 1) * c_c * (2 - c_c)
            delta_h = (hsig - 1) * c_c * (2 - c_c)
            
            # Rank-mu update
            diff = Y[:mu] - m_old
            rank_mu = np.sum([w[i] * np.outer(diff[i], diff[i]) for i in range(mu)], axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + delta_h * C) + c_mu * rank_mu
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            
            # Recompute C_sqrt and C_inv
            # Use eigendecomposition for stability
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip small negative eigenvalues due to numerical errors
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            # Check for convergence
            if sigma < 1e-12 or np.linalg.norm(m - m_old) < 1e-14:
                break
                
        return best_x, best_f
# EVOLVE-BLOCK-END
