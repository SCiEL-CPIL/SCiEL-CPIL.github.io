"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for BBOB f4 (Büche-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Standard defaults for CMA-ES
        c_sigma = (n + 1.3) / (n + 3 + 2. / n)
        c_c = (4 + n / n) / (n + 4 + 4 * n / n)
        c_1 = 2 / ((n + 1.3) ** 2 + n)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1. / (mu_eff - 2)) / ((n + 2) ** 2 + mu_eff))
        # mu is the number of weights, usually floor(n/2)
        mu = max(1, (n + 2) // 2)
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Initial state
        # Start at the center of the box
        x_mean = np.zeros(n)
        # Initial sigma: 2.0 is a good guess for [-5, 5]
        sigma = 2.0
        # Covariance matrix is identity
        C = np.eye(n)
        # Cholesky decomposition of C
        C_sqrt = np.eye(n)
        # Inverse Cholesky
        C_inv_sqrt = np.eye(n)
        
        # Evolution path
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Damping coefficients
        chi_n = np.sqrt(n) * (1 - 1. / (4 * n) + 1. / (21 * n ** 2))

        # Helper to reflect points into bounds
        def reflect(x):
            # Clip and reflect
            x_out = np.copy(x)
            # For each dimension, if out of bounds, reflect
            for i in range(n):
                if x_out[i] < -5:
                    x_out[i] = -5 + (-5 - x_out[i])
                elif x_out[i] > 5:
                    x_out[i] = 5 - (x_out[i] - 5)
            # Ensure it's strictly inside
            return np.clip(x_out, -5, 5)

        # Pre-compute inverse of C for updates? No, we update C directly.
        
        # Main Loop
        # We need to be careful with the budget. 
        # Each generation uses lambda = n + 2 (or similar) evaluations.
        # Let's use lambda = n + 2 to be standard, but ensure we don't exceed budget.
        # Actually, smaller lambda is often better for small budgets. 
        # Let's use lambda = n + 2.
        lam = n + 2
        
        # If budget is small, we might only run 1 or 2 generations.
        # 500 / 7 ~ 71 generations. That's plenty.
        
        # Initial evaluation
        x_mean = reflect(x_mean)
        f_mean = func(x_mean)
        best_x = x_mean.copy()
        best_f = f_mean
        
        # We will consume evaluations in chunks of lam.
        # To maximize efficiency and simplicity, we run generations until budget is exhausted.
        
        while budget > 0:
            # Determine population size for this generation
            # If remaining budget is less than lam, we might still run a full generation 
            # if we can afford it, or adjust lam. 
            # For simplicity and robustness, if budget < lam, we break or run partial?
            # The contract says "More than 500 calls... invalid". 
            # So we must ensure we do not exceed.
            # If budget < lam, we can reduce lam or just stop.
            # Let's just stop if we can't afford a full generation to keep CMA-ES update valid.
            if budget < lam:
                break
            
            # Sample population
            X = np.zeros((lam, n))
            F = np.zeros(lam)
            
            for i in range(lam):
                z = np.random.randn(n)
                y = x_mean + sigma * (C_sqrt @ z)
                x = reflect(y)
                X[i] = x
                F[i] = func(x)
                budget -= 1
                
            # Sort by fitness
            sort_idx = np.argsort(F)
            X = X[sort_idx]
            F = F[sort_idx]
            
            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = X[0].copy()
            
            # Update mean
            # Weighted mean of the best mu points
            x_new = np.zeros(n)
            for i in range(mu):
                x_new += weights[i] * X[i]
                
            # Update evolution path for sigma
            # Delta sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (x_new - x_mean) / sigma)
            
            # Update sigma
            h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * budget)) < (1.4 + 2 / (n + 1)) * chi_n) else 0
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update covariance matrix
            # First, update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x_mean) / sigma
            
            # Rank-mu update
            # Outer product of weighted differences
            # x_new - x_mean is the weighted mean difference
            # Actually, the rank-mu update uses the outer products of (x_i - x_mean) for the top mu individuals
            # Let's compute the rank-mu part
            # H = sum weights_i * (x_i - x_mean) (x_i - x_mean)^T
            # But standard CMA-ES uses (x_i - x_old_mean)
            # Let's stick to standard formula:
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + (1 - h_sigma) * (1 - ||p_c||^2/c_c) * C) + c_mu * sum w_i (y_i y_i^T)
            # where y_i = (x_i - x_old_mean) / sigma
            
            # Compute y_i for top mu
            Y = (X[:mu] - x_mean) / sigma
            
            # Rank-1 update part
            # p_c is already computed
            # h_sigma is computed
            # Term 1: (1 - c_1 - c_mu) * C
            C_new = (1 - c_1 - c_mu) * C
            
            # Term 2: c_1 * (p_c p_c^T + (1 - h_sigma) * (1 - ||p_c||^2/c_c) * C)
            # Note: p_c is in the "sigma-normalized" space? 
            # In standard CMA-ES, p_c is updated with (x_new - x_mean)/sigma.
            # So p_c p_c^T is the rank-1 update.
            C_new += c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * (1 - (np.dot(p_c, p_c) / c_c)) * C)
            
            # Term 3: c_mu * sum w_i * (y_i y_i^T)
            for i in range(mu):
                C_new += c_mu * weights[i] * np.outer(Y[i], Y[i])
            
            # Symmetrize to avoid numerical drift
            C = 0.5 * (C_new + C_new.T)
            
            # Update Cholesky decomposition
            # We need C_sqrt such that C = C_sqrt @ C_sqrt.T
            # And C_inv_sqrt such that C_inv_sqrt @ C @ C_inv_sqrt.T = I
            # Actually, we need C_sqrt for sampling and C_inv_sqrt for p_sigma update.
            # C_inv_sqrt is the inverse of C_sqrt.T? No.
            # If C = C_sqrt @ C_sqrt.T, then C_inv = C_sqrt^{-T} @ C_sqrt^{-1}.
            # We need C_inv_sqrt such that C_inv_sqrt @ C @ C_inv_sqrt.T = I?
            # Usually, we compute C_sqrt and then C_inv_sqrt = inv(C_sqrt.T).
            
            # Recompute Cholesky
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If not positive definite, add small jitter
                C = C + 1e-12 * np.eye(n)
                C_sqrt = np.linalg.cholesky(C)
            
            C_inv_sqrt = np.linalg.inv(C_sqrt.T)
            
            # Update mean
            x_mean = x_new
            
        return best_x, best_f

# EVOLVE-BLOCK-END
