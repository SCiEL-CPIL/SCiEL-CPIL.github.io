import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Deadline guard: 300s limit, return before 330s wall clock kill
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin
        
        dim = self.dim
        total_budget = self.budget
        
        # Strategy: Multi-start CMA-ES
        # f22 has 21 peaks. A single (1+1)-ES might get stuck.
        # We divide the budget among several independent CMA-ES runs starting from different points.
        # CMA-ES is more robust than (1+1)-ES in handling anisotropic landscapes.
        
        # Number of restarts
        # Each restart needs at least dim+1 + some iterations.
        # Let's use 5 restarts.
        n_restarts = 5
        budget_per_restart = total_budget // n_restarts
        
        global_best_x = None
        global_best_f = float('inf')
        
        # Use a fixed seed for reproducibility of the restart locations
        master_rng = np.random.default_rng(12345)
        
        for r in range(n_restarts):
            # Check time
            if time.time() > deadline:
                break
            
            # Initialize CMA-ES for this restart
            # Start point: random within [-4, 4] to stay inside box
            x0 = master_rng.uniform(-4, 4, dim)
            
            # Initial sigma
            # Box size is 10. Initial sigma should be large enough to explore.
            # CMA-ES sigma is the standard deviation of the normal distribution used for steps.
            # If we start at center, max distance to corner is ~5.7.
            # Let's start with sigma = 1.0 or slightly larger.
            sigma = 1.0
            
            # CMA-ES parameters
            # Population size lambda
            # Standard is lambda = 4 + floor(3 ln n). For n=5, 4 + 3*1.6 = 8.8 -> 9.
            # Let's use lambda = 9.
            lam = 9
            
            # CMA weights: log weights
            # mu = floor(lam / 2) = 4
            mu = lam // 2
            
            # Weights w_i proportional to log(lam/2 + 1) - log(i+1)
            weights = np.zeros(mu)
            for i in range(mu):
                weights[i] = np.log(lam/2.0 + 1) - np.log(i + 1)
            weights /= weights.sum()
            
            # Variance-effective population size
            mueff = 1.0 / np.sum(weights**2)
            
            # CMA-ES state
            # Mean
            mean = x0.copy()
            # Covariance matrix (identity)
            C = np.eye(dim)
            # Inverse square root of C (for sampling)
            # Initially C=I, so C_inv_sqrt = I
            C_inv_sqrt = np.eye(dim)
            
            # Step size control
            # Standard CMA-ES uses a learning rate for sigma
            # c_sigma = (mueff + 2) / ((n + mueff) * (4 + mueff/n))
            c_sigma = (mueff + 2) / ((dim + mueff) * (4 + mueff/dim))
            # d_sigma = 1 + 2*max(0, sqrt((mueff-1)/(dim+1)) - 1) + c_sigma
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
            
            # Covariance matrix adaptation
            # c_c = (4 - mueff/n + c_sigma) / ((n + 2)^2 + mueff)
            c_c = (4 - mueff/dim + c_sigma) / ((dim + 2)**2 + mueff)
            # c_1 = 2 / ((n + 1.3)^2 + mueff)
            c_1 = 2 / ((dim + 1.3)**2 + mueff)
            # c_mu = min(1 - c_1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)^2 + mueff))
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
            
            # Evolution path
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Calls used in this restart
            calls_used = 0
            max_calls = budget_per_restart
            
            # Initial evaluation of mean
            f_mean = func(mean)
            calls_used += 1
            
            if f_mean < global_best_f:
                global_best_f = f_mean
                global_best_x = mean.copy()
                
            # Keep track of best in this restart
            best_x_restart = mean.copy()
            best_f_restart = f_mean
            
            # Main loop for this restart
            while calls_used < max_calls:
                if time.time() > deadline:
                    break
                    
                # Generate population
                # Sample Y ~ N(0, C)
                # Y = C_inv_sqrt @ Z, Z ~ N(0, I)
                Z = master_rng.standard_normal((lam, dim))
                Y = Z @ C_inv_sqrt.T  # (lam, dim)
                
                # X = mean + sigma * Y
                X = mean + sigma * Y
                
                # Clip to box
                X = np.clip(X, -5, 5)
                
                # Evaluate
                F = np.zeros(lam)
                for i in range(lam):
                    F[i] = func(X[i])
                    calls_used += 1
                    if F[i] < global_best_f:
                        global_best_f = F[i]
                        global_best_x = X[i].copy()
                    if F[i] < best_f_restart:
                        best_f_restart = F[i]
                        best_x_restart = X[i].copy()
                        
                    # Check time inside loop for safety
                    if time.time() > deadline:
                        break
                
                if calls_used >= max_calls:
                    break
                    
                # Sort by fitness
                idx = np.argsort(F)
                X_sorted = X[idx]
                F_sorted = F[idx]
                
                # Update mean
                # w is length mu
                mean = np.sum(weights[:, np.newaxis] * X_sorted[:mu], axis=0)
                
                # Update evolution paths
                # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mueff) * C_inv_sqrt @ (mean - old_mean) / sigma
                # Note: (mean - old_mean) / sigma is the weighted sum of y_i
                # y_mean = sum(w_i * y_i)
                # y_sorted = (X_sorted - old_mean) / sigma
                # But X was clipped, so this relation is not exact.
                # Standard CMA-ES assumes no clipping or handles it carefully.
                # Given the box is large and sigma starts small, clipping is rare initially.
                # We will approximate using the unclipped Y for the path update if possible,
                # or just use the actual difference.
                # For simplicity and robustness against clipping artifacts in paths,
                # we use the difference of means divided by sigma.
                
                y_mean = (mean - (mean - sigma * np.sum(weights[:, np.newaxis] * Y[idx[:mu]], axis=0))) / sigma
                # Actually, Y[idx[:mu]] are the unclipped samples.
                # Let's use that.
                y_selected = Y[idx[:mu]]
                y_mean = np.sum(weights[:, np.newaxis] * y_selected, axis=0)
                
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * C_inv_sqrt @ y_mean
                
                # h_sigma: check if p_sigma is not too large
                # chi_n = sqrt(n) * (1 - 1/(4n) + 1/(21n^2))
                chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
                h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2*(calls_used + 1))) < (1.4 + 2/(dim + 1)) * chi_n else 0.0
                
                # p_c = (1 - c_c) * p_c + h_sigma * sqrt(c_c * (2 - c_c) * mueff) * C_inv_sqrt @ y_mean
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * C_inv_sqrt @ y_mean
                
                # Update C
                # C = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c.T + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * sum(w_i * y_i @ y_i.T)
                # Note: y_i here are the unclipped samples.
                
                # Rank-one update term
                rank_one = p_c @ p_c.T + (1 - h_sigma) * c_c * (2 - c_c) * C
                
                # Rank-mu update term
                # y_selected is (mu, dim)
                # w is (mu,)
                # weighted_outer = sum(w_i * y_i @ y_i.T)
                # This is (mu, dim) @ (dim, mu) weighted? No.
                # It is sum_{i=1..mu} w_i * y_i y_i^T
                # y_selected.T is (dim, mu)
                # y_selected is (mu, dim)
                # weighted_y = weights[:, np.newaxis] * y_selected  # (mu, dim)
                # rank_mu = weighted_y.T @ y_selected  # (dim, mu) @ (mu, dim) -> (dim, dim)
                
                weighted_y = weights[:, np.newaxis] * y_selected
                rank_mu = weighted_y.T @ y_selected
                
                C = (1 - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
                
                # Ensure C is symmetric positive definite (numerical stability)
                C = (C + C.T) / 2.0
                # Eigenvalue check? For dim=5, we can just proceed.
                # If C becomes singular, C_inv_sqrt update might fail.
                
                # Update C_inv_sqrt
                # We need to recompute C_inv_sqrt = C^{-1/2}
                # For dim=5, eigen decomposition is cheap.
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    # Clip small negative eigenvalues
                    eigvals = np.clip(eigvals, 1e-12, None)
                    C_inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
                except np.linalg.LinAlgError:
                    # Fallback: reset C to identity
                    C = np.eye(dim)
                    C_inv_sqrt = np.eye(dim)
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
                    sigma = 1.0
                    mean = best_x_restart.copy()
                    
                # Update sigma
                # sigma = sigma * exp(c_sigma / d_sigma * (||p_sigma|| / chi_n - 1))
                norm_p_sigma = np.linalg.norm(p_sigma)
                sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
                
                # Clamp sigma
                sigma = np.clip(sigma, 0.01, 10.0)
                
        return global_best_x, global_best_f
