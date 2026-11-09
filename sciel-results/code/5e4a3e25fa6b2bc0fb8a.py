"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Standard CMA-ES population size lambda = 4 + floor(3 * log(n))
        # For n=5, lambda = 4 + floor(3*1.6) = 4 + 4 = 8.
        # Let's use lambda=10 for a bit more robustness in early stages, or stick to standard.
        # Standard is often sufficient. Let's use lambda = 4 + int(3 * np.log(dim))
        lam = 4 + int(3 * np.log(dim))
        # Ensure lam is at least 2
        lam = max(2, lam)
        
        # Initial mean
        # The problem states minimizer is in [-4, 4]^5. Center is a good start.
        m = np.zeros(dim)
        
        # Initial step size
        # Heuristic: sigma0 = 1.0 or 0.5 * range / dim? 
        # Range is 10. 10/5 = 2. Let's start with 1.0.
        sigma = 1.0
        
        # Covariance matrix C, initially identity
        C = np.eye(dim)
        C_inv = np.eye(dim)
        
        # Learning rates
        # Standard values from Hansen's CMA-ES
        # mu = floor(lam / 2)
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # State variables
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Precompute Cholesky factor of C for sampling
        # Initially C=I, so Cholesky is I
        D = np.sqrt(np.diag(C))
        B = np.eye(dim)
        
        # Helper to update C, C_inv, B, D
        def update_chol(C_mat):
            # Ensure symmetry
            C_mat = (C_mat + C_mat.T) / 2
            # Add small regularization if needed for numerical stability
            C_reg = C_mat + 1e-10 * np.eye(dim)
            try:
                L = np.linalg.cholesky(C_reg)
                D_new = np.sqrt(np.diag(C_reg))
                B_new = L / D_new[:, np.newaxis]
                C_inv_new = np.linalg.inv(C_reg)
                return D_new, B_new, C_inv_new
            except np.linalg.LinAlgError:
                # Fallback if not positive definite
                C_reg += 1e-5 * np.eye(dim)
                L = np.linalg.cholesky(C_reg)
                D_new = np.sqrt(np.diag(C_reg))
                B_new = L / D_new[:, np.newaxis]
                C_inv_new = np.linalg.inv(C_reg)
                return D_new, B_new, C_inv_new
        
        calls_used = 0
        best_x = m.copy()
        best_f = float('inf')
        
        # Main loop
        while calls_used < budget:
            # Determine how many individuals we can sample
            # We need to ensure we don't exceed budget
            remaining = budget - calls_used
            if remaining < lam:
                # If we can't sample a full population, sample as many as possible
                # and break after evaluation
                num_samples = remaining
            else:
                num_samples = lam
                
            if num_samples == 0:
                break
            
            # Sample individuals
            # x = m + sigma * B * D * z
            z = np.random.randn(num_samples, dim)
            # Transform z
            # z_transformed = z @ (B.T * D) ? No.
            # x_i = m + sigma * sum_k B_kj * D_j * z_ik
            # Vectorized: X = m + sigma * (z @ (B.T * D))
            # B is dim x dim, D is dim.
            # B.T * D means scaling columns of B.T by D?
            # Let's check: B * D * z.
            # (B * D) is dim x dim.
            # z is num_samples x dim.
            # z @ (B * D).T
            BD = B * D[np.newaxis, :]
            X = m + sigma * (z @ BD.T)
            
            # Clip to box [-5, 5]
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.zeros(num_samples)
            for i in range(num_samples):
                if calls_used >= budget:
                    break
                F[i] = func(X[i])
                calls_used += 1
                
            # Update best
            for i in range(num_samples):
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = X[i].copy()
                    
            # Sort individuals by fitness
            idx = np.argsort(F)
            X_sorted = X[idx]
            F_sorted = F[idx]
            
            # We only use the best mu individuals for update
            # If we didn't sample a full population, we use all available (up to mu)
            effective_mu = min(mu, num_samples)
            if effective_mu == 0:
                break
                
            X_used = X_sorted[:effective_mu]
            w_used = w[:effective_mu]
            # Renormalize weights if we used fewer than mu
            w_used = w_used / np.sum(w_used)
            
            # New mean
            m_new = np.sum(w_used[:, np.newaxis] * X_used, axis=0)
            
            # Update step size control
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * B.T * C_inv * (m_new - m) / sigma
            # Note: m_new - m is the mean shift
            # We need to be careful with the vector direction
            
            # Calculate ps
            # term = B.T @ (C_inv @ (m_new - m))
            # Since C_inv is updated after C, we use current C_inv
            # But C is updated based on X_used.
            # Standard CMA-ES updates ps using the old B and C_inv.
            
            diff = m_new - m
            ps_new = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ (C_inv @ diff)) / sigma
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps_new) / np.sqrt(dim) - 1))
            # Cap sigma to avoid explosion or vanishing
            sigma = np.clip(sigma, 1e-12, 100)
            
            # Update pc
            pc_new = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * diff / sigma
            
            # Update Covariance C
            # Delta_h = (1 + h_sigma - c_c) * ||pc||^2 / dim - 1
            h_sigma = 1 if np.linalg.norm(ps_new) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_used / lam + 1))) < (1 + 2 / dim) * np.sqrt(dim + 2) - 1 else 0
            
            delta_h = (1 + h_sigma - c_c) * (np.linalg.norm(pc_new)**2 / dim) - 1
            
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc_new * pc_new.T + delta_h * C) + c_mu * sum(w_i * (x_i - m) * (x_i - m).T)
            
            # Calculate the rank-mu update
            # X_used is (mu, dim), w_used is (mu,)
            # diff_used = X_used - m
            diff_used = X_used - m
            rank_mu_update = np.sum(w_used[:, np.newaxis, np.newaxis] * diff_used[:, np.newaxis, :] * diff_used[:, np.newaxis, :], axis=0)
            
            C_new = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc_new, pc_new) + delta_h * C) + c_mu * rank_mu_update
            
            # Ensure symmetry
            C_new = (C_new + C_new.T) / 2
            
            # Update Cholesky and inverse
            D, B, C_inv = update_chol(C_new)
            
            # Update state
            m = m_new
            C = C_new
            pc = pc_new
            ps = ps_new
            
            # If we are close to budget, we might not do another full iteration
            # But the loop condition handles it.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
