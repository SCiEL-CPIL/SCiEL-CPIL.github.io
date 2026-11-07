"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size: standard recommendation is 4 + 3*ln(n)
        # For n=5, 4 + 3*1.6 = 8.8 -> 9
        pop_size = 9
        
        # Step size (sigma)
        sigma = 2.0
        
        # Mean (initialization)
        x = np.zeros(dim)
        
        # Covariance matrix (identity initially)
        C = np.eye(dim)
        
        # Inverse covariance matrix
        C_inv = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Learning rates
        mu = pop_size // 2  # Weights for top mu individuals
        # Standard CMA-ES weights: w_i = ln(mu+0.5) - ln(i)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # mu_eff
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Damping parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Chi_n for normalizing the norm of the step
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Initial best
        best_x = x.copy()
        best_f = func(best_x)
        
        calls_made = 1
        
        while calls_made < budget:
            # Check time limit
            if time.time() > deadline:
                break
            
            # Generate population
            # Use Cholesky of C for sampling: y = x + sigma * L @ z
            # We need L such that L @ L.T = C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite due to numerical issues, restart C
                C = np.eye(dim)
                C_inv = np.eye(dim)
                L = np.eye(dim)
            
            # Generate standard normal samples
            Z = np.random.randn(pop_size, dim)
            
            # Transform to correlated samples
            # Y = Z @ L.T (since (Z @ L.T) @ (Z @ L.T).T = Z @ L @ L.T @ Z.T = Z @ C @ Z.T? No.)
            # Correct transformation: If we want Cov(Y) = C, and Z ~ N(0, I),
            # then Y = Z @ L.T where L is Cholesky of C?
            # L @ L.T = C.
            # Cov(Z @ L.T) = L Cov(Z) L.T = L I L.T = C. Yes.
            Y = Z @ L.T
            
            # Candidates
            X = np.tile(x, (pop_size, 1)) + sigma * Y
            
            # Clip to bounds
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([func(xi) for xi in X])
            calls_made += pop_size
            
            # Track best
            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = X[idx_min].copy()
            
            # Sort by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            # Update mean
            # x_new = sum(w_i * x_{i:lambda})
            w_vec = np.zeros(pop_size)
            w_vec[:mu] = weights
            x_new = np.sum(w_vec[:, np.newaxis] * X_sorted, axis=0)
            
            # Update evolution paths
            # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mu_eff) * C_inv^{1/2} * (x_new - x) / sigma
            # C_inv^{1/2} is inverse of L
            try:
                L_inv = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                L_inv = np.eye(dim)
                
            delta_sigma = (x_new - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L_inv @ delta_sigma)
            
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-c_sigma)^{2(gen+1)}) < (1.4 + 2/(dim+1)) * chi_n
            # Approximation for h_sigma using norm ratio
            norm_p_sigma = np.linalg.norm(p_sigma)
            # We don't track generation count precisely for the damping term, use a simplified check
            # Standard implementation often just checks norm_p_sigma < (1.4 + 2/(dim+1)) * chi_n * sqrt(1 - (1-c_sigma)**2)
            # But we need the cumulative factor. Let's use a simpler heuristic or standard update.
            # Actually, CMA-ES standard uses:
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-c_sigma)^(2*t)) < (1.4 + 2/(dim+1)) * chi_n
            # Since we don't track t exactly in a simple loop without a counter, we can approximate the threshold.
            # For dim=5, (1.4 + 0.4) * chi_n ~ 1.8 * 2.2 ~ 4.
            # Let's just use the norm check against a scaled chi_n.
            h_sigma = 1.0 if norm_p_sigma < (1.4 + 2.0 / (dim + 1)) * chi_n * 1.5 else 0.0
            
            # p_c = (1 - c_c) p_c + h_sigma * sqrt(c_c (2 - c_c) mu_eff) * C_inv^{1/2} * (x_new - x) / sigma
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (L_inv @ delta_sigma)
            
            # Update covariance matrix
            # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - h_sigma^2) c_c (2 - c_c) C) + c_mu sum(w_i (y_i - y_w)(y_i - y_w)^T)
            
            # y_w = (x_new - x) / sigma
            y_w = delta_sigma
            
            # y_i - y_w for top mu
            Y_sorted = (X_sorted[:mu] - x) / sigma
            Y_diff = Y_sorted - y_w
            
            # Weighted outer sum
            # sum(w_i * (y_i - y_w) (y_i - y_w)^T)
            # This is equivalent to (Y_diff.T * w_vec[:mu]) @ Y_diff
            weighted_sum = (Y_diff.T * w_vec[:mu]) @ Y_diff
            
            # Term 1: (1 - c_1 - c_mu) C
            term1 = (1 - c_1 - c_mu) * C
            
            # Term 2: c_1 (p_c p_c^T + (1 - h_sigma^2) c_c (2 - c_c) C)
            pc_outer = np.outer(p_c, p_c)
            term2 = c_1 * (pc_outer + (1 - h_sigma**2) * c_c * (2 - c_c) * C)
            
            # Term 3: c_mu * weighted_sum
            term3 = c_mu * weighted_sum
            
            C = term1 + term2 + term3
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update C_inv
            # Instead of inverting every step (expensive), we can use Sherman-Morrison or just invert since dim=5 is small
            try:
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                C_inv = np.eye(dim)
            
            # Update x and sigma
            x = x_new
            
            # Sigma update: sigma = sigma * exp(c_sigma / chi_n * (||p_sigma|| / chi_n - 1))
            sigma = sigma * np.exp(c_sigma / chi_n * (norm_p_sigma / chi_n - 1))
            
            # Clamp sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-4, 10.0)
            
            # Check for convergence or stagnation
            if sigma < 1e-4:
                break
                
            # If we used all budget, the loop condition handles it
            # But we need to ensure we don't exceed budget
            if calls_made >= budget:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
