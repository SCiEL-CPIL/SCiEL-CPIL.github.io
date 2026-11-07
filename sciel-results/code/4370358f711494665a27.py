import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin of 50s
        
        d = self.dim
        budget = self.budget
        
        # CMA-ES parameters tuned for small budget in low dimensions
        lambda_pop = 10  # Population size
        mu = lambda_pop // 2  # Selection size
        
        # Weights for the mean update
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        
        # Step size control parameters
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))
        
        # Initial state
        x_mean = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        C_sqrt = np.eye(d)
        
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls = 1
        
        # Expected norm of N(0, I)
        c_n = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
        
        while calls + lambda_pop <= budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate population
            z = np.random.randn(lambda_pop, d)
            X = x_mean + sigma * (z @ C_sqrt.T)
            
            # Clip to bounds
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            calls += lambda_pop
            
            # Update best
            idx_best = np.argmin(F)
            if F[idx_best] < best_f:
                best_f = F[idx_best]
                best_x = X[idx_best].copy()
                
            # Selection
            idx_sort = np.argsort(F)
            X_selected = X[idx_sort[:mu]]
            
            # Mean update
            x_mean_old = x_mean.copy()
            x_mean = x_mean + sigma * (weights @ X_selected)
            
            # Sigma update
            step = (x_mean - x_mean_old) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * step
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / c_n - 1))
            
            # Covariance Matrix update
            p_c = (1 - c_c) * p_c + c_c * step
            
            Y = (X_selected - x_mean_old) / sigma
            
            C = (1 - c_1 - c_mu + mu_eff * c_mu) * C
            C += c_1 * np.outer(p_c, p_c)
            C += c_mu * (Y.T @ (Y * weights[:, None]))
            
            # Ensure symmetry
            C = (C + C.T) / 2.0
            
            # Update C_sqrt
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If Cholesky fails, add jitter to C
                C += 1e-4 * np.eye(d)
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Reset if still failing
                    C_sqrt = np.eye(d)
                    C = np.eye(d)
                    sigma *= 0.5
        
        return best_x, best_f
