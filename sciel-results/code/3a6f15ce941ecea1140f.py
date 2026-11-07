"""Rewritten optimizer for BBOB-style f23 (Katsuura) using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 270  # safety margin
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        def cma_es(x_init, sigma_init, max_calls, rng, deadline):
            nonlocal calls_made, best_x, best_f
            
            # CMA-ES Parameters for small dim
            N = 4 + 3 * int(np.floor(np.log(dim)))  # Population size, approx 10 for dim=5
            N = max(N, 10)
            
            # Weights
            w = np.log(N/2 + 0.5) - np.log(np.arange(1, N + 1))
            w /= w.sum()
            mu = int(np.sum(w > 0))
            w = w[:mu]
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w**2)
            
            # Adaptation rates
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / (dim**1.6 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / (dim**2 + mu_eff))
            
            # State
            m = x_init.copy()
            sigma = sigma_init
            C = np.eye(dim)
            C_inv = np.eye(dim)
            C_sqrt = np.eye(dim)
            C_sqrt_inv = np.eye(dim)
            
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            def update_chol(C):
                nonlocal C_sqrt, C_sqrt_inv
                try:
                    L = np.linalg.cholesky(C)
                    C_sqrt = L
                    C_sqrt_inv = np.linalg.inv(L)
                    return True
                except np.linalg.LinAlgError:
                    # Regularize
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    C = (C + C.T) / 2
                    L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
                    C_sqrt = L
                    C_sqrt_inv = np.linalg.inv(L)
                    return True

            update_chol(C)
            
            calls_in_cma = 0
            last_best_f = np.inf
            stall_count = 0
            
            while calls_in_cma < max_calls and time.time() < deadline:
                # Generate population
                ys = rng.multivariate_normal(np.zeros(dim), C, size=N)
                # Actually, using standard normal and transforming is safer/faster
                zs = rng.standard_normal((N, dim))
                xs = np.array([m + sigma * (C_sqrt @ z.T).T for z in zs])
                
                # Evaluate
                fs = np.array([func(x) for x in xs])
                calls_in_cma += N
                calls_made += N
                
                # Update global best
                idx_min = np.argmin(fs)
                if fs[idx_min] < best_f:
                    best_f = fs[idx_min]
                    best_x = xs[idx_min].copy()
                    
                if best_f < last_best_f - 1e-10:
                    last_best_f = best_f
                    stall_count = 0
                else:
                    stall_count += 1
                    
                # Stop if stalled too long or sigma too small
                if stall_count > 5 or sigma < 1e-8:
                    break
                    
                # Sort by fitness
                order = np.argsort(fs)
                xs_sorted = xs[order]
                fs_sorted = fs[order]
                
                # Update mean
                m_old = m
                m = np.sum(w[:, None] * xs_sorted[:mu], axis=0)
                
                # Update path and sigma
                # p_sigma = (1-c_sigma) p_sigma + c_sigma * sqrt(mu_eff) * C_inv_sqrt * (m - m_old) / sigma
                diff = (m - m_old) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ diff)
                
                norm_p_sigma = np.linalg.norm(p_sigma)
                chi_N = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
                sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_N - 1))
                
                # Update path for covariance
                # p_c = (1-c_c) p_c + c_c * sqrt(mu_eff) * C_inv_sqrt * (m - m_old)
                # But we need to check if p_sigma was in the update direction
                if norm_p_sigma / chi_N < 1.4 + 2/(dim+1):
                    p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_sqrt_inv @ diff)
                    
                # Update covariance
                # C = (1 - c_mu - c_1 + c_1*delta) C + c_mu * sum w_i (z_i - rho z)(z_i - rho z)^T + c_1 * (p_c p_c^T - (1-c_1)C)
                # Simplified: Use the standard rank-mu update
                zs_sorted = zs[order]
                z_mu = np.sum(w[:, None] * zs_sorted[:mu], axis=0)
                
                # Correction for bias
                # rho_z = 2 * (dim/2 + 1) - dim
                # This is getting complex. Let's use a simplified covariance update often used in compact implementations.
                
                # Rank-1 update part
                h_sigma = 1 if norm_p_sigma / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_in_cma + 1) / (dim * c_sigma))) < (1.4 + 2 / (dim + 1)) * chi_N else 0
                # This h_sigma calculation is approximate. Let's simplify.
                
                # Simpler Cov Update:
                # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - h_sigma)(1 - c_1) C) + c_mu sum w_i (z_i - rho z)(z_i - rho z)^T
                # Let's just do the rank-mu update on the transformed space and transform back?
                # No, let's stick to the standard formula.
                
                # To avoid bugs in complex CMA-ES implementation, let's use a simpler, robust variant:
                # Adapt only Sigma and Mean, and use a fixed covariance? No, that's (1+1)-ES.
                # Let's try a simplified CMA-ES where we update C using the outer product of the best steps.
                
                # Actually, for D=5, we can just do:
                # C = 0.99 C + 0.01 * (1/mu_eff) * sum (x_i - m)(x_i - m)^T / sigma^2
                # This is a simplified adaptation.
                
                # Let's use the proper update but carefully.
                # z_diffs = zs_sorted[:mu] - np.sum(w * zs_sorted[:mu], axis=0)
                # C_update = np.sum(w[:, None, None] * np.einsum('ij,ik->ijk', z_diffs, z_diffs), axis=0)
                
                # Let's use a very stable simplified version:
                # 1. Update Mean
                # 2. Update Sigma
                # 3. Update Covariance using the covariance of the successful steps
                successful_zs = zs_sorted[:mu]
                mean_z = np.mean(successful_zs, axis=0) # Unweighted mean for simplicity? No, weighted.
                mean_z = np.sum(w[:, None] * successful_zs, axis=0)
                
                # Center the successful zs
                centered_zs = successful_zs - mean_z
                # Weighted covariance
                C_new = (1 - c_1 - c_mu) * C + c_mu * np.einsum('i,ij,ik->jk', w, centered_zs, centered_zs)
                # Add rank-1 from p_c
                C_new += c_1 * (np.outer(p_c, p_c) - (1 - c_1) * C)
                
                C = C_new
                update_chol(C)

            return best_x, best_f, calls_in_cma

        # Phase 1: Global CMA-ES
        x0 = rng.uniform(-5, 5, dim)
        sigma0 = 2.0
        budget_phase1 = 300
        _, _, used1 = cma_es(x0, sigma0, budget_phase1, rng, deadline)
        
        # Phase 2: Local CMA-ES around best
        if best_x is not None and calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            x1 = best_x.copy()
            sigma1 = 0.5
            _, _, used2 = cma_es(x1, sigma1, remaining, rng, deadline)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
