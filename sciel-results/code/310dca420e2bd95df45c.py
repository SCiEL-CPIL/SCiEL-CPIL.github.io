import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 290.0  # Safety margin before 330s kill
        
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        N = 4 + int(3 * np.log(n)) if n > 1 else 2
        # For n=5, N=8. This is a standard choice.
        
        # Initial mean
        x = np.zeros(n)
        
        # Initial step size. 
        # The search space is [-5, 5]. The minimum is in [-4, 4].
        # A sigma of 2.5 covers ~10 units (4 sigma), which is good for initial exploration.
        sigma = 2.5
        
        # Covariance matrix
        C = np.eye(n)
        
        # Cumulation paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Learning rates
        mu = N // 2  # Number of weighted points
        
        # Weights
        # Standard log-linear weights
        w = np.log(mu + 1) - np.log(1 + np.arange(mu))
        w = w / np.sum(w)
        
        mu_eff = 1 / np.sum(w**2)
        
        # Learning rates for step size and covariance
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        # Damping factors
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        # Cholesky factor of C (C = B @ B.T)
        B = np.linalg.cholesky(C)
        
        best_x = x.copy()
        best_f = float('inf')
        calls = 0
        
        # Initial evaluation
        x_init = np.clip(x, -5, 5)
        f_init = func(x_init)
        calls += 1
        if f_init < best_f:
            best_f = f_init
            best_x = x_init.copy()
            
        gen = 0
        
        while calls < budget:
            if time.time() - start_time > timeout:
                break
                
            # Generate population
            offspring = np.zeros((N, n))
            fitness = np.zeros(N)
            z_vectors = np.zeros((N, n))
            
            for i in range(N):
                if calls >= budget or time.time() - start_time > timeout:
                    break
                    
                z = np.random.standard_normal(n)
                z_vectors[i] = z
                y = x + sigma * B @ z
                y = np.clip(y, -5, 5)
                
                f_y = func(y)
                calls += 1
                offspring[i] = y
                fitness[i] = f_y
                
                if f_y < best_f:
                    best_f = f_y
                    best_x = y.copy()
            
            if calls >= budget or time.time() - start_time > timeout:
                break
            
            # Sort offspring by fitness
            order = np.argsort(fitness)
            y_sorted = offspring[order]
            z_sorted = z_vectors[order]
            
            # Update mean vector
            x_new = np.sum(w[:, np.newaxis] * y_sorted[:mu], axis=0)
            
            # Step size adaptation (CSA)
            # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mu_eff) * B^-1 (y_1 - x) / sigma
            # Note: y_1 - x = sigma B z_1, so B^-1 (y_1 - x) / sigma = z_1
            # This simplification is valid because we stored z.
            y_1 = y_sorted[0]
            z_1 = z_sorted[0]
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_1
            
            # Check condition for h_sigma
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1 - c_sigma)^{2(gen+1)}) < (1.4 + 2/(n+1)) chi_n
            # We need to track the decay of the norm. 
            # Approximation: use the current norm and a threshold.
            # A more robust way is to track the decay factor explicitly.
            # For simplicity and speed in d=5, we can use a standard approximation or just check the norm.
            # Let's use the standard threshold check with a tracked decay term.
            # Actually, the standard implementation tracks `decay` or uses the formula directly.
            # Here, we approximate the threshold.
            # The term sqrt(1 - (1 - c_sigma)^{2(gen+1)}) approaches 1.
            # For early generations, it is smaller.
            # Let's just use the norm check against the threshold.
            threshold_sigma = (1.4 + 2.0 / (n + 1)) * chi_n
            # If we don't track the exact decay, we might trigger h_sigma incorrectly.
            # However, for d=5, this is less critical.
            # Let's implement the decay tracking properly for correctness.
            # decay_sigma = 1 - (1 - c_sigma)**(2 * (gen + 1))
            # But we don't have gen easily accessible in a way that tracks the exact power without storing it.
            # Let's store the decay factor.
            # Actually, I can just compute it.
            decay_sigma = 1 - (1 - c_sigma) ** (2 * (gen + 1))
            if np.linalg.norm(p_sigma) / np.sqrt(decay_sigma) < threshold_sigma:
                h_sigma = 1.0
            else:
                h_sigma = 0.0
            
            # Update p_c
            # p_c = (1 - c_c) p_c + h_sigma * sqrt(c_c (2 - c_c) mu_eff) * z_1
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * z_1
            
            # Check condition for h_c
            # h_c = 1 if ||p_c|| / sqrt(1 - (1 - c_c)^{2(gen+1)}) < (1.4 + 2/(n+1)) chi_n
            decay_c = 1 - (1 - c_c) ** (2 * (gen + 1))
            if np.linalg.norm(p_c) / np.sqrt(decay_c) < threshold_sigma:
                h_c = 1.0
            else:
                h_c = 0.0
            
            # Update Covariance Matrix C
            # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - h_c) c_c (2 - c_c) C) + c_mu sum w_i (y_i - x)(y_i - x)^T
            # Note: (y_i - x) = sigma B z_i
            # So (y_i - x)(y_i - x)^T = sigma^2 B z_i z_i^T B^T
            # This is expensive to compute directly if we don't use the z vectors.
            # But we have z_sorted.
            # Let's use the formula with z vectors to avoid matrix multiplications with B if possible?
            # The standard update is:
            # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - h_c) c_c (2 - c_c) C) 
            #   + c_mu sigma^2 B (sum w_i z_i z_i^T) B^T
            
            # Compute the weighted sum of z z^T
            # z_sorted[:mu] is (mu, n)
            # w is (mu,)
            # We want sum w_i z_i z_i^T
            # This is Z_w @ Z.T where Z_w = w[:, None] * Z
            Z_w = w[:, np.newaxis] * z_sorted[:mu]
            rank_mu_update = Z_w @ z_sorted[:mu].T
            
            # Update C
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(p_c, p_c) + (1 - h_c) * c_c * (2 - c_c) * C)
            C += c_mu * (sigma ** 2) * (B @ rank_mu_update @ B.T)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2.0
            # Add a small regularization if needed, but CMA-ES usually handles it.
            # If Cholesky fails, add identity.
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-10 * np.eye(n)
                B = np.linalg.cholesky(C)
            
            # Update mean
            x = x_new
            
            # Adapt sigma
            # sigma = sigma * exp(c_sigma / chi_n * (||p_sigma|| / chi_n - 1))
            sigma = sigma * np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1.0))
            
            # Ensure sigma doesn't get too small or too large
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            gen += 1
        
        return best_x, best_f
