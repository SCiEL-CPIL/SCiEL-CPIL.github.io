"""Direct approach: CMA-ES tuned for Bent Cigar (f12) with 1/5 success rule."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES optimizer for Bent Cigar (f12).
    Adapts the search distribution (covariance) to the anisotropic landscape.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2  # Number of best individuals to update weights
        
        # Weights for best mu individuals
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Step size
        sigma = 2.5  # Initial step size, slightly larger to cover box initially
        
        # Mean
        m = np.zeros(dim)
        
        # Covariance Matrix
        C = np.eye(dim)
        
        # Inverse square root of C for sampling
        # We use eigendecomposition or Cholesky. For small dim, eig is fine and robust.
        def update_inv_sqrt_C(C):
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip small negative values due to numerical errors
            eigvals = np.clip(eigvals, 1e-16, None)
            inv_sqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            return inv_sqrt_C

        # Pre-compute initial
        inv_sqrt_C = update_inv_sqrt_C(C)
        
        # Evolution path for sigma (cs) and C (cs)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        
        # State
        ps = np.zeros(dim)  # Evolution path for sigma
        pc = np.zeros(dim)  # Evolution path for C
        
        # Track best
        best_x = m.copy()
        best_fx = float('inf')
        calls_used = 0
        
        # Main Loop
        while calls_used + lam <= budget:
            if time.time() >= deadline:
                break
            
            # Generate population
            # Sample from N(m, sigma^2 * C)
            # z ~ N(0, I)
            Z = np.random.default_rng(42).standard_normal((lam, dim))
            Y = m + sigma * (Z @ inv_sqrt_C.T)
            
            # Evaluate
            F = np.array([func(np.clip(y, -5, 5)) for y in Y])
            calls_used += lam
            
            # Update best
            idx_min = np.argmin(F)
            if F[idx_min] < best_fx:
                best_fx = F[idx_min]
                best_x = Y[idx_min].copy()
            
            # Sort individuals
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]
            
            # Update mean
            m_old = m.copy()
            m = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)
            
            # Update evolution path for sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (inv_sqrt_C @ (m - m_old) / sigma)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_used + lam) / dim))) < (1 + 2 / (dim + 1)) * np.sqrt(1 - (1 - cs) ** (2 * (calls_used + lam) / dim))
            
            # Update evolution path for C
            pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mu_eff) * (m - m_old) / sigma
            
            # Update Covariance Matrix
            delta_hsig = (1 - hsig) * cs * (2 - cs)
            rank1_update = (1 + delta_hsig - c1) * np.outer(pc, pc)
            
            rankmu_update = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (Y_sorted[i] - m_old) / sigma
                rankmu_update += weights[i] * np.outer(y_i, y_i)
            
            C = (1 + c1 * delta_hsig - c1 - cmu) * C + c1 * rank1_update + cmu * rankmu_update
            
            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_used + lam) / dim)) - 1))
            
            # Update inverse square root of C
            # Periodically or every step? Every step is expensive but dim=5 is small.
            # To save time, we can do it every few steps, but let's try every step first.
            inv_sqrt_C = update_inv_sqrt_C(C)
            
            # Early stop if sigma is tiny or convergence
            if sigma < 1e-10:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
