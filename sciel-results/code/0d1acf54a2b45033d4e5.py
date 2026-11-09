"""Rewritten program using CMA-ES for better performance on multimodal/ill-conditioned functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer with time-based budget management."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size lambda
        lam = max(4, int(4 + 3 * np.log(dim)))
        # Weighting
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size adaptation parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial state
        # Start at center, but slightly randomized to avoid symmetry issues if needed
        # However, for BBOB, the optimum is shifted. A uniform start is safer.
        # But CMA-ES is usually started near the optimum. Since we don't know it,
        # we start at 0, which is the center of the box [-5, 5].
        x = np.zeros(dim)
        
        # Initial sigma
        # Box is [-5, 5], so width 10. 
        # A reasonable initial sigma is roughly (max - min) / 4
        sigma = 2.0
        
        # Covariance matrix
        C = np.eye(dim)
        # Inverse square root of C
        C_inv_sqrt = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial evaluation
        # We need to be careful with the budget.
        # Each generation costs lam evaluations.
        # We have 'budget' total evaluations.
        
        best_x = x
        best_f = func(x)
        calls = 1
        
        # Main loop
        while calls + lam <= budget:
            # Check time
            if time.time() > deadline - 1.0:
                break
            
            # Sample points
            ys = np.zeros((lam, dim))
            for i in range(lam):
                z = np.random.standard_normal(dim)
                # Cholesky decomposition of C for sampling
                # We update C every generation, so we need to decompose it.
                # For efficiency, we can cache the decomposition or just do it here.
                # Since dim=5, this is cheap.
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Regularize if not positive definite
                    C += 1e-10 * np.eye(dim)
                    L = np.linalg.cholesky(C)
                ys[i] = x + sigma * (L @ z)
                
                # Clip to box
                ys[i] = np.clip(ys[i], -5, 5)
            
            # Evaluate
            fs = np.array([func(y) for y in ys])
            calls += lam
            
            # Update best
            idx_min = np.argmin(fs)
            if fs[idx_min] < best_f:
                best_f = fs[idx_min]
                best_x = ys[idx_min].copy()
            
            # Sort
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]
            
            # Mean update
            x_old = x.copy()
            x_new = np.sum(weights[:, None] * ys[:mu], axis=0)
            
            # Step size adaptation (CSA)
            # Calculate norm of (x_new - x_old)
            dx = x_new - x_old
            norm_dx = np.linalg.norm(dx)
            if norm_dx > 1e-16:
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ dx) / sigma
                h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (dim + 1)) * np.trace(np.eye(dim)) else 0
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(np.trace(np.eye(dim))) - 1))
            else:
                h_sigma = 1
            
            # Covariance matrix adaptation
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * dx / sigma
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(ys[i] - x_old, ys[i] - x_old)
            
            C = (1 + c_1 * h_sigma * (dim + 1) - c_1 - c_mu) * C + c_1 * (rank_one - (1 - h_sigma) * (1 - c_c) * C) + c_mu / sigma**2 * rank_mu
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Update C_inv_sqrt
            try:
                # Eigen decomposition for stable square root
                vals, vecs = np.linalg.eigh(C)
                # Clip small negative eigenvalues due to numerical error
                vals = np.clip(vals, 1e-16, None)
                C_inv_sqrt = (vecs / np.sqrt(vals)) @ vecs.T
            except np.linalg.LinAlgError:
                C += 1e-10 * np.eye(dim)
                try:
                    vals, vecs = np.linalg.eigh(C)
                    vals = np.clip(vals, 1e-16, None)
                    C_inv_sqrt = (vecs / np.sqrt(vals)) @ vecs.T
                except:
                    C_inv_sqrt = np.eye(dim)
            
            x = x_new
            
            # Termination check: if sigma is very small or variance is low
            if sigma < 1e-12:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
