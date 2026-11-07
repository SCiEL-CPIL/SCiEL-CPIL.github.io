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
        deadline = start_time + 280
        
        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0
        
        # Multi-start CMA-ES with progressively smaller initial sigmas
        # For Sphere (convex), large sigma finds basin, small sigma refines
        sigma_schedule = [4.0, 1.0, 0.2, 0.05]
        
        for init_sigma in sigma_schedule:
            if calls >= budget or time.time() > deadline:
                break
            
            # CMA-ES Parameters
            lambda_ = int(4 + 3 * np.log(n))
            mu = lambda_ // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1 / np.sum(weights**2)
            
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
            
            # Start from current best or center
            if np.isinf(best_f):
                x_mean = np.zeros(n)
            else:
                x_mean = best_x.copy()
            
            sigma = init_sigma
            C = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            
            def update_sqrtC(C_mat):
                w, V = np.linalg.eigh(C_mat)
                w = np.maximum(w, 1e-12)
                sqrtC = V @ np.diag(np.sqrt(w)) @ V.T
                inv_sqrtC = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
                return sqrtC, inv_sqrtC
            
            sqrtC, inv_sqrtC = update_sqrtC(C)
            
            # Evaluate starting point
            x0 = x_mean.copy()
            f0 = func(x0)
            calls += 1
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            
            # Stagnation tracking
            last_improvement_call = calls
            stagnation_window = 50
            
            # Main loop for this start
            while calls < budget:
                if time.time() > deadline:
                    break
                
                # Check stagnation
                if calls - last_improvement_call >= stagnation_window:
                    break
                
                # Generate population
                Y = np.random.randn(lambda_, n)
                X = np.array([x_mean + sigma * (Y[i] @ sqrtC) for i in range(lambda_)])
                
                F = np.zeros(lambda_)
                for i in range(lambda_):
                    if calls >= budget or time.time() > deadline:
                        break
                    F[i] = func(X[i])
                    calls += 1
                
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
                    last_improvement_call = calls
                
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
                
                C = (1 + c_c * (2 - c_c) * mu_eff - c_1 - c_mu) * C \
                    + c_1 * (p_c @ p_c - C) \
                    + c_mu * sum(weights[i] * (X[i] - x_mean) @ (X[i] - x_mean).T for i in range(mu)) / sigma**2
                C = (C + C.T) / 2
                sqrtC, inv_sqrtC = update_sqrtC(C)
                
                if sigma < 1e-12:
                    break
            
            # If we still have budget, try next smaller sigma
            if calls < budget and time.time() < deadline:
                continue
            else:
                break
        
        return best_x, best_f
