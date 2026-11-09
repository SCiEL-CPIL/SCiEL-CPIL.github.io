"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for continuous black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # CMA-ES parameters
        # Population size
        lam = 12
        
        # Initial mean: center of the box [-5, 5]
        x_mean = np.zeros(n)
        
        # Initial standard deviation
        sigma = 1.5
        
        # Covariance matrix
        C = np.eye(n)
        
        # Weights for selection (mu best individuals)
        mu = lam // 2
        weights = np.arange(1, mu + 1) ** 1.5
        weights = weights / weights.sum()
        
        # Variance-effective selection size
        mueff = 1.0 / np.sum(weights**2)
        
        # Step size control parameters
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * ((mueff + 0.5) / (n + 2) - 1))
        
        # Boundary handling parameters
        lower_bound = -5.0
        upper_bound = 5.0
        
        # State variables for CMA-ES
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls_used = 1
        
        while calls_used < budget:
            # 1. Sampling
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                L = np.eye(n)
                
            # Generate random samples
            z = rng.standard_normal((lam, n))
            # Transform to y
            y = z @ L.T * sigma
            # Generate candidates
            x_samples = x_mean + y
            
            # Clip to bounds
            x_samples = np.clip(x_samples, lower_bound, upper_bound)
            
            # Evaluate candidates
            f_values = np.zeros(lam)
            for i in range(lam):
                if calls_used >= budget:
                    break
                f_values[i] = func(x_samples[i])
                calls_used += 1
                if f_values[i] < best_f:
                    best_f = f_values[i]
                    best_x = x_samples[i].copy()
            
            # If we ran out of budget, break
            if calls_used >= budget:
                break
                
            # 2. Selection
            # Sort indices by fitness
            sorted_idx = np.argsort(f_values)
            # Take the best mu
            selected_idx = sorted_idx[:mu]
            x_selected = x_samples[selected_idx]
            
            # 3. Update mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum(weights[:, np.newaxis] * x_selected, axis=0)
            
            # 4. Update covariance matrix
            # Evolution path p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (
                (x_mean - x_mean_old) / (sigma * np.sqrt(np.diag(C)))
            )
            
            # h_sigma check
            norm_p_sigma = np.linalg.norm(p_sigma)
            # Approximation for large iterations
            h_sigma = 1.0 if norm_p_sigma / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (n + 1)) * c_sigma else 0.0
            
            # p_c update
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (
                (x_mean - x_mean_old) / sigma
            )
            
            # C update
            # Rank-1 part
            rank1 = p_c * p_c.T + (1 - h_sigma) * c_c * (2 - c_c) * np.diag(C)
            
            # Rank-mu part
            y_selected = (x_selected - x_mean_old) / sigma
            rank_mu = np.sum(weights[:, np.newaxis, np.newaxis] * (y_selected[:, :, np.newaxis] * y_selected[:, np.newaxis, :]), axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * rank1 + c_mu * rank_mu
            
            # Ensure C is symmetric and positive definite (numerical stability)
            C = (C + C.T) / 2
            
            # Sigma update
            expected_norm = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / expected_norm - 1))
            
            # Clip sigma to reasonable range
            sigma = np.clip(sigma, 1e-6, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
