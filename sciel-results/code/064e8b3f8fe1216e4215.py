import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 290  # Safety margin
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # For dim=5, standard CMA-ES parameters
        lambda_ = 4 + int(np.floor(3 * np.log(dim)))  # Population size
        mu = lambda_ // 2  # Number of weights
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Strategy parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial state
        # Start at 0, which is usually a decent guess or quickly moved away from
        m = np.zeros(dim)
        sigma = 2.0  # Initial step size, covering a good chunk of [-5, 5]
        
        # Covariance matrix and its square root (via Cholesky)
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = m.copy()
        best_fx = func(best_x)
        calls_used = 1
        
        # Main loop
        while calls_used < budget and time.time() < deadline:
            # Generate population
            ys = []
            xs = []
            for i in range(lambda_):
                z = np.random.randn(dim)
                y = C_sqrt @ z
                x = m + sigma * y
                # Clip to bounds to be safe, though func also clips
                x = np.clip(x, -5, 5)
                ys.append(y)
                xs.append(x)
            
            # Evaluate
            fs = []
            for x in xs:
                f_val = func(x)
                fs.append(f_val)
                calls_used += 1
                if f_val < best_fx:
                    best_fx = f_val
                    best_x = x.copy()
            
            # Sort by fitness
            sort_indices = np.argsort(fs)
            ys = [ys[i] for i in sort_indices]
            xs = [xs[i] for i in sort_indices]
            fs = [fs[i] for i in sort_indices]
            
            # Update mean
            m_new = m + sigma * sum(weights[i] * ys[i] for i in range(mu))
            
            # Update evolution path p_sigma
            # Need C^{-1/2}
            # For small dim, we can invert C directly or use Cholesky
            # C is symmetric positive definite
            try:
                L = np.linalg.cholesky(C)
                L_inv = np.linalg.inv(L)
                C_inv_sqrt = L_inv.T
            except np.linalg.LinAlgError:
                # Fallback if Cholesky fails (shouldn't happen often with proper updates)
                C_inv_sqrt = np.linalg.inv(C)
                C_inv_sqrt = C_inv_sqrt.T @ C_inv_sqrt
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (m_new - m) / sigma
            
            # Check for divergence
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_used // lambda_))) > (1.4 + 2 / (dim + 1)) * np.sqrt(dim):
                sigma /= 1.5
                # Reset p_sigma if it's too large
                p_sigma = np.zeros(dim)
            
            # Update evolution path p_c
            # C_inv_sqrt is already computed
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt @ (m_new - m) / sigma
            
            # Update covariance matrix
            # Rank-one update
            delta_sig = (np.linalg.norm(p_sigma)**2 - 1) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_used // lambda_)))
            C = (1 + c_1 * delta_sig) * C + c_1 * np.outer(p_c, p_c)
            
            # Rank-mu update
            # Sum of weighted outer products of y
            y_weighted = np.zeros((dim, mu))
            for i in range(mu):
                y_weighted[:, i] = weights[i] * ys[i]
            
            # C update: C = (1 - c_1 - c_mu) * C + c_1 * outer(p_c, p_c) + c_mu * sum(w_i * outer(y_i, y_i))
            # Note: The standard formula is C = (1 - c_1 - c_mu) * C + c_1 * outer(p_c, p_c) + c_mu * Y Y^T
            # Where Y Y^T is the weighted sum of outer products
            
            # Recalculate C more carefully
            C = (1 + c_1 * delta_sig - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * (y_weighted @ y_weighted.T)
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update Cholesky decomposition
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add small identity to make it positive definite
                C += 1e-10 * np.eye(dim)
                C_sqrt = np.linalg.cholesky(C)
            
            m = m_new
            # Clip mean to bounds to keep search centered
            m = np.clip(m, -5, 5)
            
        return best_x, best_fx
