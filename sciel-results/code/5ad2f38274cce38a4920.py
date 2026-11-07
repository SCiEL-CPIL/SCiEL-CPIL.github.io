import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        
        # CMA-ES Parameters
        lambda_ = 10
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation rates
        c_sigma = (1 + 2 * max(0, np.sqrt((lambda_ - 2) / (lambda_ + 2)) - 1)) / (lambda_ + 1.3)
        c_c = (4 - lambda_ + 1) / (lambda_ + 1.4 + 4 - lambda_)
        c_1 = 2 / ((lambda_ + 1.7)**2 + dim)
        c_mu = 2 * (mu_eff - 2 + 1 / mu_eff) / ((lambda_ + 1.7)**2 + dim) - c_1
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 2) / (dim + 2)) - 1) + c_sigma
        
        # Initialization
        # Start near center or random? Random is safer for unknown shift.
        # But Gallagher peaks are spread. A random start in [-5,5] is standard.
        rng = np.random.default_rng(42)
        m = rng.uniform(-5, 5, dim)
        sigma = 2.0
        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Precompute square root of C for sampling? 
        # We need to update C and its square root.
        # For dim=5, we can just use eigendecomposition every step or maintain C and use cholesky/eigh for sampling.
        # Standard CMA-ES uses C and updates it. Sampling requires sqrt(C).
        # Since dim is small (5), computing eigh every generation is cheap.
        
        best_x = m
        best_f = float('inf')
        
        # Initial population evaluation
        # We will generate lambda points per generation.
        # Budget is 500. Lambda=10. Generations = 50.
        
        for gen in range(budget // lambda_):
            if time.time() - start_time > 250: # Safety margin
                break
                
            # Sample lambda points
            # y = z, x = m + sigma * C_sqrt @ z
            # We need C_sqrt.
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            eigvals = np.maximum(eigvals, 1e-12)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            Z = rng.standard_normal((lambda_, dim))
            X = m[None, :] + sigma * (Z @ C_sqrt.T)
            
            # Clip to bounds
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            
            # Track best
            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = X[idx_min].copy()
            
            # Sort
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            # Update mean
            w = weights
            m_old = m
            m = np.sum(w[:, None] * X_sorted[:mu], axis=0)
            
            # Update covariance
            # H = (1/sigma) * (X_sorted[:mu] - m_old).T
            H = (X_sorted[:mu] - m_old)[None, :, :].T
            H = H / sigma
            
            # Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c * p_c.T + (1 - (1 - c_c) * (1 - c_c)) * C) + c_mu * sum(w_i * h_i * h_i.T)
            
            # Update p_c
            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff * c_c * (2 - c_c) / dim) * (m - m_old) / sigma
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv @ (m - m_old) / sigma
            
            # Update sigma
            # chi_n = sqrt(n) * (1 - 1/(2*n^2) + 1/(8*n^3))
            chi_n = np.sqrt(dim) * (1 - 1.0 / (2 * dim**2) + 1.0 / (8 * dim**3))
            sigma *= np.exp((d_sigma / chi_n) * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update C
            # Need C_sqrt_inv for p_sigma update? 
            # In standard CMA-ES, p_sigma is updated using C^{-1/2}.
            # We have C_sqrt, so C_sqrt_inv = inv(C_sqrt).
            C_sqrt_inv = np.linalg.inv(C_sqrt)
            
            # Recalculate p_sigma properly? 
            # The formula above used C_sqrt_inv which was from the previous step's C. 
            # This is an approximation. For dim=5 and 50 gens, it's likely fine.
            # Let's stick to the standard update.
            
            # H^T W H
            W = np.diag(w[:mu])
            HWH = H.T @ W @ H
            
            # Rank-1 update part
            # The term (1 - (1 - c_c)^2) * C is part of the rank-1 update if we expand?
            # Standard formula:
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + (1 - (1 - c_c)^2) * C) + c_mu * HWH
            # Simplifies to:
            # C = (1 - c_1 - c_mu) * C + c_1 * p_c p_c^T + c_mu * HWH
            # Wait, the term $(1 - (1 - c_c)^2) C$ is usually included in the rank-1 update to ensure positive definiteness?
            # Actually, the standard update is:
            # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - (1 - c_c)^2) C) + c_mu \sum w_i h_i h_i^T
            # This can be rewritten as:
            # C = (1 - c_1 - c_mu + c_1(1 - (1 - c_c)^2)) C + c_1 p_c p_c^T + c_mu HWH
            # Let's use the explicit form to be safe.
            
            rank1 = p_c[:, None] * p_c[None, :]
            rank1 += (1 - (1 - c_c)**2) * C
            
            C = (1 - c_1 - c_mu) * C + c_1 * rank1 + c_mu * HWH
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update C_sqrt_inv for next iteration? 
            # We recompute it at the start of the loop.
            
        return best_x, best_f
