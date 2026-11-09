"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin

        rng = np.random.default_rng(42)

        # CMA-ES Parameters
        # For dim=5, lambda=4*5+4=24 is a good default.
        # mu = lambda // 2 = 12
        lambda_ = max(4 * dim + 4, 10)
        mu_ = lambda_ // 2
        
        # Weights
        w = np.zeros(mu_)
        for i in range(mu_):
            w[i] = np.log(mu_ + 0.5) - np.log(i + 1)
        w /= w.sum()
        
        mu_eff = 1.0 / np.sum(w ** 2)
        
        # Strategy parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * min(mu_ / (dim + 2) - 1, 1) + c_sigma
        c_c = (4 + dim / c_sigma) / (dim + 4 + 2 / c_sigma)
        c_1 = 2 / (dim ** 2 + 6)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / (dim ** 2 + 2))
        
        # State
        x_mean = np.zeros(dim)
        sigma = 2.0  # Initial step size, domain is [-5, 5]
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Eigen decomposition cache
        D = np.ones(dim)
        B = np.eye(dim)
        
        # Pre-compute B and D from C (initially I)
        eigvals, eigvecs = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(eigvals, 1e-12))
        B = eigvecs
        
        # Track best
        best_x = x_mean.copy()
        best_f = func(best_x)
        call_count = 1
        
        # Chi_n constant for sigma update
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        while call_count < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            offspring = []
            for i in range(lambda_):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                f_y = func(y)
                call_count += 1
                offspring.append((y, f_y, z))
                
                if f_y < best_f:
                    best_f = f_y
                    best_x = y.copy()
            
            if len(offspring) < mu_:
                break
                
            # Sort by fitness
            offspring.sort(key=lambda x: x[1])
            selected = offspring[:mu_]
            
            # Update mean
            old_mean = x_mean.copy()
            x_mean = np.sum([w[i] * y for i, (y, f, z) in enumerate(selected)], axis=0)
            
            # Update evolution paths
            # C^{-1/2} (x_new - x_old)
            # We need C^{-1/2}. We have B and D such that C = B D^2 B^T.
            # C^{-1/2} = B D^{-1} B^T
            C_inv_sqrt = B @ np.diag(1.0 / np.maximum(D, 1e-10)) @ B.T
            delta_mean = x_mean - old_mean
            
            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * (C_inv_sqrt @ delta_mean)
            
            # Update sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * np.sum([w[i] * z for i, (y, f, z) in enumerate(selected)], axis=0)
            
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update Covariance Matrix
            # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - (1 - c_sigma)^2) I) + c_mu sum w_i (y_i - x_mean)(y_i - x_mean)^T
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_sigma) ** 2) * np.eye(dim))
            
            for i in range(mu_):
                diff = selected[i][0] - x_mean
                C += c_mu * w[i] * np.outer(diff, diff)
            
            C = (C + C.T) / 2  # Symmetrize
            
            # Ensure positive semi-definite
            try:
                eigvals_C, _ = np.linalg.eigh(C)
                if np.min(eigvals_C) < 1e-12:
                    C = C + (1e-10 - np.min(eigvals_C)) * np.eye(dim)
            except:
                C = np.eye(dim)
                
            # Update eigen decomposition for next iteration
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(eigvals, 1e-12))
                B = eigvecs
            except:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
