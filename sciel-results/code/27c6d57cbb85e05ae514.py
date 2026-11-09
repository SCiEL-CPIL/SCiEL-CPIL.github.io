"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB f13 (Sharp ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        
        # Parameters
        x0 = np.zeros(dim)
        sigma0 = 2.0
        popsize = 4 + int(3 * np.log(dim))  # lambda = 7 for dim=5
        mu = popsize // 2  # number of best points
        w = np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1 / (w @ w)
        
        # CMA-ES parameters
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        c_sigma = (mu_eff + 5) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # State
        x = x0.copy()
        sigma = sigma0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        best_x = x0.copy()
        best_f = func(best_x)
        calls = 1
        
        # Pre-compute w column vector for outer products
        w_col = w[:, np.newaxis]
        
        while calls < budget:
            # Check time limit with safety margin
            if time.time() - t0 > 290:
                break
                
            # 1. Sampling
            # Compute eigendecomposition of C for sampling
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            sqrt_eigvals = np.sqrt(eigvals)
            
            Y = np.zeros((popsize, dim))
            for i in range(popsize):
                z = np.random.randn(dim)
                # x + sigma * C^0.5 * z
                Y[i] = x + sigma * (eigvecs @ (sqrt_eigvals * z))
                Y[i] = np.clip(Y[i], -5, 5)
            
            # 2. Evaluation
            F = np.zeros(popsize)
            for i in range(popsize):
                if calls >= budget:
                    # If we run out of budget mid-evaluation, fill with best known
                    F[i] = best_f 
                    break
                F[i] = func(Y[i])
                calls += 1
            
            # 3. Selection and Update
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]
            
            if F[0] < best_f:
                best_f = F[0]
                best_x = Y[0].copy()
            
            # Weighted mean
            m = (Y[:mu] * w_col).sum(axis=0)
            
            # Update paths
            # Need C_old^-0.5. We have eigvecs and eigvals from C_old (before update).
            inv_sqrt_eigvals = 1 / sqrt_eigvals
            C_old_inv_sqrt = (eigvecs * inv_sqrt_eigvals) @ eigvecs.T
            
            ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(mu_eff) / sigma * (C_old_inv_sqrt @ (m - x))
            pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) * (m - x) / sigma
            
            # Update covariance
            # C = (1 - c_1 - c_mu) C + c_1 (pc pc^T + (1 - c_c) c_c I) + c_mu sum w_i (y_i - m)(y_i - m)^T
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(pc, pc) + (1 - c_c) * c_c * np.eye(dim))
            
            # Efficient update for c_mu term
            # sum w_i (y_i - m)(y_i - m)^T
            # Can be computed as W @ (Y_diff @ Y_diff.T) where W is diagonal? No.
            # It is sum_i w_i * outer(Y[i]-m, Y[i]-m)
            # For small mu and dim, direct loop is fine.
            diff = Y[:mu] - m
            C += c_mu * (diff.T * w) @ diff
            
            # Update mean and sigma
            x = m
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Ensure C remains positive definite (symmetrize and clip eigenvalues if necessary)
            C = (C + C.T) / 2.0
            
        return best_x, best_f
# EVOLVE-BLOCK-END
