import numpy as np

class Optimizer:
    """CMA-ES continuous optimizer for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))
        # Selection size (half of population)
        mu = lam // 2
        
        # Weights for the selected points
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial values
        x0 = np.zeros(dim)
        sigma = 2.0 # Initial step size
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        
        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial mean
        m = x0.copy()
        
        best_x = m.copy()
        best_f = float('inf')
        calls = 0
        
        # Pre-compute Cholesky factors
        def update_chol(C):
            nonlocal B, D
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive definite
            eigvals = np.maximum(eigvals, 1e-20)
            D = np.sqrt(eigvals)
            B = eigvecs
            return B, D

        update_chol(C)
        
        while calls < budget:
            # Generate population
            Y = []
            X = []
            F = []
            
            # Determine how many points we can evaluate in this generation
            # We want to evaluate lam points, but we might not have budget
            # If budget < lam, we just evaluate what we can? 
            # Actually, CMA-ES needs a full population for the update step.
            # If we don't have enough budget for a full population, we should just do random search or finish.
            # With budget 500 and lam 13, we have ~38 generations.
            
            if budget - calls < lam:
                # Not enough budget for a full generation
                # Evaluate remaining calls randomly around best
                for _ in range(budget - calls):
                    x = m + sigma * np.random.randn(dim)
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    calls += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                break
            
            # Sample lam points
            for i in range(lam):
                z = np.random.randn(dim)
                x = m + sigma * (B @ (D * z))
                x = np.clip(x, -5, 5)
                Y.append(z)
                X.append(x)
                fx = func(x)
                F.append(fx)
                calls += 1
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            Y = np.array(Y)
            X = np.array(X)
            F = np.array(F)
            
            # Sort by fitness
            idx = np.argsort(F)
            X = X[idx]
            Y = Y[idx]
            F = F[idx]
            
            # Update mean
            m_old = m.copy()
            m = np.sum(weights[:, np.newaxis] * X[:mu], axis=0)
            
            # Update sigma
            z = (B.T @ (D * Y[:mu].T))
            # Weighted sum of Y
            y_sigma = np.sum(weights[:, np.newaxis] * Y[:mu], axis=0)
            
            # Correction term
            h_sigma = 1.0
            # Threshold for h_sigma
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            
            # Calculate p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            
            # Check condition for h_sigma
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls / lam))) < (1.4 + 2 / (dim + 1)) * chi_n:
                h_sigma = 1.0
            else:
                h_sigma = 0.0
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            # Clip sigma to reasonable range
            sigma = np.clip(sigma, 1e-12, 10.0)
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (B @ (D * y_sigma))
            
            # Update Covariance Matrix C
            # Rank-1 update
            v = (m - m_old) / sigma
            C = (1 + c_1 * h_sigma * (mu_eff - 1) - c_1) * C + c_1 * np.outer(v, v)
            
            # Rank-mu update
            Y_mu = Y[:mu]
            # Weighted covariance
            C_rank_mu = np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(Y_mu.T, Y_mu.T), axis=0)
            # Note: The standard formula uses Y, which are the standardized samples.
            # However, the update is usually:
            # C = (1 + c_1 * h_sigma * (mu_eff - 1) - c_1) * C + c_1 * v v^T + c_mu * sum w_i (y_i y_i^T)
            # But the Y in the update should be the standardized ones.
            # Let's stick to the standard CMA-ES update.
            
            # Wait, the standard update uses the actual sampled Y (standard normals).
            # The term is c_mu * sum_{i=1}^mu w_i * y_i y_i^T
            # We already have Y[:mu] which are the standardized normals.
            
            C = (1 + c_1 * h_sigma * (mu_eff - 1) - c_1 - c_mu) * C + c_1 * np.outer(v, v) + c_mu * np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(Y[:mu].T, Y[:mu].T), axis=0)
            
            # Ensure positive definite
            # Periodically re-orthonormalize
            if calls % lam == 0:
                update_chol(C)
                # Check for numerical issues
                if not np.all(np.isfinite(C)):
                    C = np.eye(dim)
                    B = np.eye(dim)
                    D = np.ones(dim)
                    sigma = 2.0
                    m = best_x.copy()
                    
        return best_x, best_f
