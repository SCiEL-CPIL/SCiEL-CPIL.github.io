"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        n = self.dim
        # Population size
        lambda_pop = 4 + int(3 * np.log(n))
        # Number of selected parents
        mu = lambda_pop // 2
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        # Variance-effective selection size
        mueff = 1.0 / np.sum(weights**2)
        
        # Initial parameters
        m = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        
        # Learning rates
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, (mueff - 1) / (n + 1) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * ((mueff + 2) / (n + 2) - 1))
        
        # Cumulation paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Cholesky factors
        B = np.eye(n)
        D = np.ones(n)
        
        best_x = m.copy()
        best_f = float('inf')
        calls = 0
        
        while calls < self.budget:
            if time.time() > deadline:
                break
            
            # Generate population
            ys = []
            for _ in range(lambda_pop):
                z = np.random.randn(n)
                y = B @ (D * z)
                x = m + sigma * y
                x = np.clip(x, -5, 5)
                fy = func(x)
                calls += 1
                ys.append((fy, x))
                
                if fy < best_f:
                    best_f = fy
                    best_x = x.copy()
                    
            # Sort by fitness
            ys.sort(key=lambda t: t[0])
            
            # Update mean
            m_new = np.sum([weights[i] * ys[i][1] for i in range(mu)], axis=0)
            
            # Update covariance
            diff = np.array([ys[i][1] - m for i in range(mu)])
            
            # Step size update
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (B @ (D * (m_new - m) / sigma))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
            h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls + 1))) < (1.4 + 2 / (n + 1)) * chi_n) else 0
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Covariance update
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (m_new - m) / sigma
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C)
            for i in range(mu):
                C += c_mu * weights[i] * np.outer(diff[i], diff[i])
                
            # Ensure positive definiteness
            try:
                B = np.linalg.cholesky(C)
                D = 1.0 / np.sqrt(np.diag(B))
            except np.linalg.LinAlgError:
                C = C + 1e-6 * np.eye(n)
                try:
                    B = np.linalg.cholesky(C)
                    D = 1.0 / np.sqrt(np.diag(B))
                except np.linalg.LinAlgError:
                    B = np.eye(n)
                    D = np.ones(n)
            
            m = m_new
            
        return best_x, best_f
# EVOLVE-BLOCK-END
