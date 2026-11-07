import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lambda_ = int(4 + 3 * np.log(n))
        # Weights
        mu = lambda_ // 2  # Number of points used for mean update
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights**2)
        
        # Step size adaptation
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        
        # Covariance matrix adaptation
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Initial step size
        # The box is [-5, 5], center is 0. Min is in [-4, 4].
        # Start with a sigma that covers a good portion of the space.
        sigma = 2.0
        
        # Initial mean
        x_mean = np.zeros(n)
        
        # Initial covariance matrix
        C = np.eye(n)
        
        # Evolution path for sigma
        p_sigma = np.zeros(n)
        # Evolution path for covariance
        p_c = np.zeros(n)
        
        # Precompute sqrt(C) and C^-1/2
        def update_sqrtC():
            nonlocal C, sqrtC, inv_sqrtC
            # Eigen decomposition
            w, V = np.linalg.eigh(C)
            w = np.maximum(w, 1e-12)  # Ensure positive
            sqrtC = V @ np.diag(np.sqrt(w)) @ V.T
            inv_sqrtC = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
            return sqrtC, inv_sqrtC

        sqrtC, inv_sqrtC = update_sqrtC()
        
        # Initial evaluation
        x0 = x_mean.copy()
        f0 = func(x0)
        calls = 1
        best_x = x0.copy()
        best_f = f0
        
        # Main loop
        while calls < budget:
            if time.time() - start_time > 280:
                break
                
            # Generate population
            Y = np.random.randn(lambda_, n)
            X = np.array([x_mean + sigma * (Y[i] @ sqrtC) for i in range(lambda_)])
            
            # Clip to box (function does it, but we do it to keep state clean if we used it)
            # The task says func(x) clips. We just call it.
            
            F = np.zeros(lambda_)
            for i in range(lambda_):
                F[i] = func(X[i])
                calls += 1
                if calls >= budget:
                    break
                if time.time() - start_time > 280:
                    break
                    
            if calls >= budget:
                break
                
            # Sort by function value
            indices = np.argsort(F)
            X = X[indices]
            F = F[indices]
            
            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = X[0].copy()
            
            # Update mean
            y_w = np.zeros(n)
            for i in range(mu):
                y_w += weights[i] * (X[i] - x_mean) / sigma
            x_mean = x_mean + sigma * y_w
            
            # Update sigma
            y_sigma = inv_sqrtC @ y_w
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
            
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update covariance
            y_c = inv_sqrtC @ y_w
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
            
            # Rank-one and rank-mu updates
            C = (1 + c_c * (2 - c_c) * mu_eff - c_1 - c_mu) * C \
                + c_1 * (p_c @ p_c - C) \
                + c_mu * sum(weights[i] * (X[i] - x_mean) @ (X[i] - x_mean).T for i in range(mu)) / sigma**2
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Update sqrt(C)
            sqrtC, inv_sqrtC = update_sqrtC()
            
            # Optional: Check for convergence
            if sigma < 1e-12:
                break

        return best_x, best_f
