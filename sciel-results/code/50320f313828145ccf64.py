import numpy as np
import time

class Optimizer:
    """CMA-ES Optimizer for BBOB f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 290  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # For dim=5 and budget=500, a small population is efficient.
        lam = 4  # Population size
        mu = lam // 2  # Number of best points for recombination
        
        # Weights for recombination
        # Standard weights: w_i = log(mu + 0.5) - log(i)
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= np.sum(weights)
        
        # Variance-effective selection mass
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Adaptation rates
        cs = (mueff + 1.7) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        
        # Initial state
        # Start at the center of the box, with a large enough sigma to explore
        m = np.zeros(dim)
        sigma = 3.0  # Large initial step to cover [-5, 5] roughly
        
        # Covariance matrix
        C = np.eye(dim)
        # Square root of C (for sampling)
        C_sqrt = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Track best solution
        best_x = m.copy()
        best_f = func(m)
        calls_used = 1
        
        # Main loop
        while calls_used < budget:
            if time.time() > deadline:
                break
            
            # 1. Sample offspring
            offspring = np.zeros((lam, dim))
            for i in range(lam):
                z = np.random.randn(dim)
                # Sample from N(0, C) using C_sqrt
                # m + sigma * C_sqrt @ z
                offspring[i] = m + sigma * (C_sqrt @ z)
                # Clip to bounds
                offspring[i] = np.clip(offspring[i], -5, 5)
            
            # 2. Evaluate offspring
            offspring_f = np.zeros(lam)
            for i in range(lam):
                if time.time() > deadline:
                    break
                offspring_f[i] = func(offspring[i])
                calls_used += 1
                if offspring_f[i] < best_f:
                    best_f = offspring_f[i]
                    best_x = offspring[i].copy()
            
            if calls_used >= budget or time.time() > deadline:
                break
                
            # 3. Sort offspring by fitness
            sorted_indices = np.argsort(offspring_f)
            offspring = offspring[sorted_indices]
            offspring_f = offspring_f[sorted_indices]
            
            # 4. Update mean
            # Weighted average of the best mu points
            m_new = np.sum(offspring[:mu] * weights[:, np.newaxis], axis=0)
            
            # 5. Update evolution paths
            # p_sigma
            # h_sigma: check if p_sigma is not too large
            # chi_n = E[||N(0, I)||] = sqrt(n) * (1 - 1/(4n) + 1/(32n^2))
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(32*dim**2))
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - sigma) ** (2 / budget)) < (1.4 + 2/(dim+1)) * chi_n else 0
            
            p_sigma = (1 - cs) * p_sigma + h_sigma * np.sqrt(cs * (2 - cs) * mueff) * (C_sqrt.T @ ((m_new - m) / sigma))
            
            # p_c
            h_c = 1 if np.linalg.norm(p_c) / np.sqrt(1 - (1 - cs) ** (2 / budget)) < (1.4 + cs) * chi_n else 0
            p_c = (1 - cs) * p_c + h_c * np.sqrt(cs * (2 - cs) * mueff) * (m_new - m) / sigma
            
            # 6. Update covariance matrix
            # C = (1 - c1 - cmu) * C + c1 * (p_c p_c^T + (1 - h_c) * (1 - cs^2) * C) + cmu * sum(w_i * (y_i y_i^T))
            # where y_i = (x_i - m_old) / sigma
            y = (offspring[:mu] - m) / sigma
            rank_mu = np.sum(y * weights[:, np.newaxis], axis=0)[:, np.newaxis] @ y.T
            
            C = (1 - c1 - cmu) * C + c1 * (np.outer(p_c, p_c) + (1 - h_c) * (1 - cs**2) * C) + cmu * rank_mu
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Eigenvalue decomposition to update C_sqrt and ensure PD
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip small negative eigenvalues to 0
                eigvals = np.clip(eigvals, 1e-10, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
            except np.linalg.LinAlgError:
                # Fallback: reset C to identity if decomposition fails
                C = np.eye(dim)
                C_sqrt = np.eye(dim)
            
            # 7. Update step size sigma
            # 1/5 success rule
            # sigma = sigma * exp(cs / ds * (||p_sigma|| / chi_n - 1))
            sigma = sigma * np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Keep sigma within reasonable bounds
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Update mean
            m = m_new
        
        return best_x, best_f
