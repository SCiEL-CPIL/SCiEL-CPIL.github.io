import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin
        
        D = self.dim
        # Standard CMA-ES parameters
        lambda_pop = 4 + int(3 * np.log(D))
        mu = lambda_pop // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Step size adaptation parameters
        c_sigma = (mueff + 2) / (D + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (D + 1)) - 1) + c_sigma
        
        # Covariance matrix adaptation parameters
        c_c = (4 + mueff / D) / (D + 4 + 2 * mueff / D)
        c_1 = 2 / ((D + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((D + 2) ** 2 + mueff))
        
        # Initial state
        # Start near the center, as the minimizer is in [-4, 4]^5
        x_mean = np.zeros(D)
        sigma = 2.0
        
        # Path accumulators
        p_sigma = np.zeros(D)
        p_c = np.zeros(D)
        
        # Covariance matrix
        C = np.eye(D)
        # Cholesky decomposition for sampling
        try:
            C_sqrt = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C_sqrt = np.eye(D)
        
        # Pre-compute some constants
        chi_n = np.sqrt(D) * (1 - 1 / (4 * D) + 1 / (21 * D ** 2))
        
        best_x = x_mean.copy()
        best_f = np.inf
        calls_made = 0
        
        while calls_made < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate population
            # We need to generate lambda_pop candidates
            # To be efficient, we generate all at once if possible, but we must stop if budget runs out mid-generation
            # Since lambda_pop is small (8 for D=5), we can just loop or vectorize carefully.
            # Vectorizing is faster.
            
            # Check if we have enough budget for a full generation
            if self.budget - calls_made < lambda_pop:
                # If not, we might just do one more evaluation or break.
                # Let's just break to keep it simple, or do individual evals.
                # Actually, CMA-ES needs the full set for ranking. 
                # If we can't do a full generation, we can just stop or do a (1+1) step.
                # Given D=5, lambda=8. If we have < 8 calls left, we can't do a full CMA-ES step.
                # We can fall back to (1+1)-ES for the remaining calls.
                for _ in range(self.budget - calls_made):
                    if time.time() > deadline: break
                    y = x_mean + sigma * np.linalg.multi_dot([C_sqrt, np.random.randn(D)])
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                    if fy < func(x_mean): # Simple (1+1) update logic if we were tracking current mean f
                         # This is getting complex. Just stopping is safer and 8 calls is small.
                         pass
                break
            
            # Generate full population
            Z = np.random.randn(lambda_pop, D)
            Y = x_mean[None, :] + sigma * (Z @ C_sqrt.T)
            Y = np.clip(Y, -5, 5)
            
            # Evaluate population
            F = np.array([func(y) for y in Y])
            calls_made += lambda_pop
            
            # Update best
            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = Y[idx_min].copy()
            
            # Sort by fitness
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]
            
            # Compute new mean
            y_w = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)
            
            # Update sigma (CSA)
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (C^-1/2 * (y_w - x_mean))
            # Note: C^-1/2 is the inverse of the Cholesky factor
            try:
                C_inv_sqrt = np.linalg.inv(C_sqrt)
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(D)
                
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (C_inv_sqrt @ (y_w - x_mean))
            
            if np.linalg.norm(p_sigma) / chi_n < (1.4 + 2 / (D + 1)) * (1 - c_sigma):
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update covariance matrix
            # p_c = (1 - c_c) * p_c + sqrt(c_c * (2 - c_c) * mueff) * (C^-1/2 * (y_w - x_mean))
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * (C_inv_sqrt @ (y_w - x_mean))
            
            # Rank-one update
            C = (1 + c_c * (2 - c_c) - c_1 - c_mu) * C \
                + c_1 * (p_c[:, None] * p_c[None, :] + (c_c * (2 - c_c) - c_1) * np.eye(D)) \
                + c_mu * sum(weights[i] * (Y_sorted[i] - x_mean)[:, None] * (Y_sorted[i] - x_mean)[None, :] for i in range(mu))
            
            # Regularize C to ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            
            # Update Cholesky
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback if C becomes non-PD
                C = np.eye(D)
                C_sqrt = np.eye(D)
            
            # Update mean
            x_mean = y_w
        
        return best_x, best_f
