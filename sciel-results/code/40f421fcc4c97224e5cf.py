"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.linalg

class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) implementation.
    Uses population and covariance adaptation to handle rotated landscapes.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        
        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(dim))
        if lam < 2:
            lam = 2
            
        sigma = 2.0
        mean = np.zeros(dim)
        C = np.eye(dim)
        
        # Eigen decomposition for efficient sampling and updates
        D_sq, B = np.linalg.eigh(C)
        D = np.sqrt(D_sq)
        
        mu = lam // 2
        w = np.log(np.arange(1, mu + 1)) - np.log(mu)
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = mean.copy()
        best_f = func(best_x)
        calls = 1
        
        start_time = time.time()
        deadline = start_time + 290.0
        
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Sample population
            z = rng.standard_normal((lam, dim))
            # y = mean + sigma * B * D * z
            y = mean + sigma * (B @ (D * z.T).T).T
            
            fvals = np.zeros(lam)
            for i in range(lam):
                if calls >= budget:
                    break
                y_i = np.clip(y[i], -5, 5)
                fvals[i] = func(y_i)
                calls += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = y_i.copy()
            
            if calls < budget + lam - 1: # If we broke early due to budget, fvals has uninitialized parts
                 # We need to ensure we only sort the evaluated ones
                 pass
            
            # If we didn't complete the generation, we can't update CMA properly.
            # Simple strategy: if budget runs out mid-generation, just break.
            # The loop condition `calls < budget` ensures we stop before exceeding.
            # But if we break inside the for loop, we haven't evaluated all lam points.
            # We should check if we have lam points.
            if calls - 1 < lam: 
                 # We didn't complete the generation
                 break

            order = np.argsort(fvals)
            y_sorted = y[order]
            
            mean_new = mean + sum(w[i] * (y_sorted[i] - mean) for i in range(mu))
            
            # Compute C_inv_sqrt
            # Use eigendecomposition: C = B D^2 B^T
            # C_inv_sqrt = B D^{-1} B^T
            D_inv = 1.0 / np.maximum(D, 1e-12)
            C_inv_sqrt = B @ np.diag(D_inv) @ B.T
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (mean_new - mean))
            
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) - np.sqrt(dim)))
            
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ (mean_new - mean))
            
            chi_N = np.sqrt(dim * (1 - 1.0/(4*dim) - 1.0/(21*dim**2)))
            h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2/(dim+1)) * chi_N) else 0
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C \
                + c_1 * (p_c @ p_c.T + (1 - h_sigma) * c_c * (2 - c_c) * C) \
                + c_mu * sum(w[i] * (y_sorted[i] - mean) @ (y_sorted[i] - mean).T for i in range(mu))
            
            # Update eigen decomposition
            try:
                D_sq, B = np.linalg.eigh(C)
                D_sq = np.maximum(D_sq, 1e-12)
                D = np.sqrt(D_sq)
            except Exception:
                C = np.eye(dim)
                D = np.ones(dim)
                B = np.eye(dim)
                
            mean = mean_new
            
        return best_x, best_f
# EVOLVE-BLOCK-END
