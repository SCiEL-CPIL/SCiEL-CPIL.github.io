import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s. We stop at 290s to be safe.
        timeout = 290.0
        
        n = self.dim
        # Population size: Standard CMA-ES recommendation
        # For n=5, N = 4 + floor(3 * ln(5)) = 4 + 3 = 7. 
        # Using N=8 for slightly better statistical stability.
        N = 8
        
        # CMA-ES Parameters
        # Initial mean: Center of the box
        x = np.zeros(n)
        # Initial step size: 
        # Box is [-5, 5]. Min is in [-4, 4]. 
        # Sigma = 1.5 allows ~3 sigma = 4.5 coverage, fitting well in the box.
        sigma = 1.5
        
        # Covariance matrix
        C = np.eye(n)
        
        # Cumulation paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Learning rates
        # mu: number of recombination weights (top half)
        mu = N // 2
        
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(1 + np.arange(mu))
        weights = weights / np.sum(weights)
        
        # Effective population size
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size adaptation (CSA)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        
        # Covariance matrix adaptation
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Damping factors
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        # Cholesky factor of C
        # C = B @ B.T
        B = np.linalg.cholesky(C)
        
        best_x = x.copy()
        best_f = float('inf')
        
        calls = 0
        budget = self.budget
        gen = 0
        
        # Initial evaluation
        x_init = np.clip(x, -5, 5)
        f_init = func(x_init)
        calls += 1
        if f_init < best_f:
            best_f = f_init
            best_x = x_init.copy()
            
        while calls < budget:
            if time.time() - start_time > timeout:
                break
                
            # Generate population
            offspring = np.zeros((N, n))
            fitness = np.zeros(N)
            z_vectors = np.zeros((N, n))
            
            for i in range(N):
                if time.time() - start_time > timeout or calls >= budget:
                    break
                    
                # Sample standard normal
                z = np.random.standard_normal(n)
                z_vectors[i] = z
                
                # Transform to y
                # y = x + sigma * B @ z
                y = x + sigma * (B @ z)
                
                # Clip to box
                y = np.clip(y, -5, 5)
                
                # Evaluate
                f = func(y)
                calls += 1
                
                offspring[i] = y
                fitness[i] = f
                
                if f < best_f:
                    best_f = f
                    best_x = y.copy()
                    
            if calls >= budget or time.time() - start_time > timeout:
                break
                
            # Sort offspring by fitness
            order = np.argsort(fitness)
            y_sorted = offspring[order]
            z_sorted = z_vectors[order]
            
            # Update mean vector
            x_new = np.sum(weights[:, np.newaxis] * y_sorted[:mu], axis=0)
            
            # Step size adaptation (CSA)
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * B^{-1} (y_1 - x) / sigma
            # Note: B^{-1} (y_1 - x) / sigma is approximately z_1 if C is well-conditioned, 
            # but strictly we need to solve the linear system.
            # y_1 - x = sigma * B @ z_1
            # So B^{-1} (y_1 - x) / sigma = z_1
            # We stored z_sorted, so z_1 is z_sorted[0]
            
            delta_sigma = z_sorted[0]
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * delta_sigma
            
            # h_sigma condition
            norm_p_sigma = np.linalg.norm(p_sigma)
            # Threshold for h_sigma
            # chi_n * sqrt(1 - (1 - c_sigma)^(2 * gen))
            # For large gen, (1 - c_sigma)^(2 * gen) -> 0
            # We use a simplified check: if norm is small enough
            # Standard condition: ||p_sigma|| / sqrt(1 - (1 - c_sigma)^(2g)) < (1.4 + 2/(n+1)) * chi_n
            # Approximation for large g: ||p_sigma|| < (1.4 + 2/(n+1)) * chi_n
            threshold_sigma = (1.4 + 2.0 / (n + 1)) * chi_n
            h_sigma = 1 if norm_p_sigma < threshold_sigma else 0
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (norm_p_sigma / chi_n - 1))
            
            # Update cumulative path for covariance (p_c)
            # p_c = (1 - c_c) * p_c + h_sigma * sqrt(c_c * (2 - c_c) * mu_eff) * B^{-1} (y_1 - x) / sigma
            # Again, B^{-1} (y_1 - x) / sigma = z_1
            delta_c = z_sorted[0]
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * delta_c
            
            # h_c condition
            norm_p_c = np.linalg.norm(p_c)
            # Threshold for h_c
            # (1 + 2/(n+1)) * chi_n * sqrt(1 - (1 - c_c)^(2 * gen))
            # Approximation: (1 + 2/(n+1)) * chi_n
            threshold_c = (1 + 2.0 / (n + 1)) * chi_n
            h_c = 1 if norm_p_c**2 < (1 + 2.0 / (n + 1)) * chi_n**2 * n else 0
            
            # Update Covariance Matrix C
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + (1 - h_c) * c_c * (2 - c_c) * C) 
            #     + c_mu * sum_{i=1}^{mu} w_i * (y_i - x)(y_i - x)^T
            # Note: (y_i - x) = sigma * B @ z_i
            # So (y_i - x)(y_i - x)^T = sigma^2 * B @ z_i z_i^T @ B^T
            # This form is more stable if we update B directly, but let's stick to C update for simplicity, 
            # then recompute B.
            
            # Rank-1 update
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_c) * c_c * (2 - c_c) * C)
            
            # Rank-mu update
            # Sum of weighted outer products of z vectors, scaled by sigma^2 and transformed by B
            # We need to compute sum(w_i * z_i z_i^T)
            z_weighted = z_sorted[:mu] * np.sqrt(weights)[:, np.newaxis]
            # sum of outer products
            # Using matrix multiplication for efficiency: (z_weighted.T @ z_weighted) gives sum(z_i z_i^T) * w_i? No.
            # z_weighted is mu x n. z_weighted.T @ z_weighted is n x n.
            # Element (j,k) = sum_i (z_weighted[i,j] * z_weighted[i,k]) = sum_i (sqrt(w_i) z_i[j] * sqrt(w_i) z_i[k]) = sum_i w_i z_i[j] z_i[k]
            rank_mu_matrix = z_weighted.T @ z_weighted
            
            # C += c_mu * sigma^2 * B @ rank_mu_matrix @ B^T
            C += c_mu * (sigma**2) * (B @ rank_mu_matrix @ B.T)
            
            # Ensure C is positive definite (symmetrize and add small jitter if needed)
            C = (C + C.T) / 2.0
            
            # Update Cholesky factor
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C becomes non-positive definite, add small diagonal
                C += 1e-10 * np.eye(n)
                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Reset C to identity if it fails badly
                    C = np.eye(n)
                    B = np.eye(n)
            
            # Update mean
            x = x_new
            gen += 1
            
        return best_x, best_f
