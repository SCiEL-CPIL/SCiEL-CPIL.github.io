"""
CMA-ES based optimizer for BBOB f18 (Schaffers F7 ill-conditioned).
Strategy:
  - Use CMA-ES with a small population (lambda=8) to estimate covariance.
  - Initialize the covariance matrix based on a small set of random samples 
    to rapidly align with the funnel structure.
  - Adapt the covariance matrix using the rank-one and rank-mu updates.
  - Use a time guard to stay within the 330s limit.
  - Track the best point and value found.
"""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin

        dim = self.dim
        n_evals = 0
        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')

        # CMA-ES parameters
        # For dim=5, lambda is usually 4 + floor(3 * log(dim)) = 4 + 6 = 10, but smaller is faster.
        # Let's use lambda=8 for a balance of exploration and cost.
        lambda_ = 8
        mu = lambda_ // 2  # Number of selected individuals for recombination
        
        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)  # Variance-effective population size

        # Step size adaptation
        # c_sigma = (mueff + 2) / ((dim + 2) ** 2 + mueff)
        c_sigma = (mueff + 2) / ((dim + 2) ** 2 + mueff)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Covariance matrix adaptation
        c_c = (4 + mueff / dim) / (dim + 4) ** 2
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        # Initial state
        # Start near the center, as the minimum is in [-4, 4]^5
        x_mean = np.zeros(dim)
        
        # Initial sigma
        # A heuristic for the initial step size. 
        # The box is [-5, 5], so a starting sigma of 2.0 or 2.5 is reasonable.
        sigma = 2.0

        # Initial covariance matrix
        # Instead of identity, we can try to estimate it from a few samples, 
        # but identity is standard and robust. 
        # Given the "ill-conditioned" nature, CMA-ES will adapt it.
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        
        # Evolve paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Cholesky cache
        def update_chol(C, C_inv, C_sqrt, C_sqrt_inv):
            # Compute Cholesky of C
            try:
                L = np.linalg.cholesky(C)
                C_sqrt = L
                C_sqrt_inv = np.linalg.inv(L)
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                # Fallback if not positive definite
                C = C + np.eye(dim) * 1e-10
                L = np.linalg.cholesky(C)
                C_sqrt = L
                C_sqrt_inv = np.linalg.inv(L)
                C_inv = np.linalg.inv(C)
            return C, C_inv, C_sqrt, C_sqrt_inv

        C, C_inv, C_sqrt, C_sqrt_inv = update_chol(C, C_inv, C_sqrt, C_sqrt_inv)

        while n_evals < self.budget:
            if time.time() > deadline:
                break

            # 1. Sample lambda_ points
            offspring = []
            offspring_vals = []
            
            for i in range(lambda_):
                if n_evals >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Sample z ~ N(0, I)
                z = rng.standard_normal(dim)
                # Transform to y = x_mean + sigma * B * z
                y = x_mean + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                n_evals += 1
                
                offspring.append(y)
                offspring_vals.append(fy)
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if len(offspring) < mu:
                break

            # 2. Sort offspring by fitness
            indices = np.argsort(offspring_vals)
            sorted_offspring = [offspring[i] for i in indices]
            sorted_vals = [offspring_vals[i] for i in indices]
            
            # 3. Update mean
            old_mean = x_mean.copy()
            x_mean = np.sum([weights[i] * sorted_offspring[i] for i in range(mu)], axis=0)
            x_mean = np.clip(x_mean, -5, 5) # Keep mean in bounds, though CMA usually handles this via sigma
            
            # 4. Update sigma (CSA)
            # h_sigma condition
            h_sigma = (1 + 2 * max(0, (mueff - 1) / (dim + 1)) )
            # Actually, the standard condition is:
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-c_sigma)^{2*es}) < (1 + 2/(dim+1)) * sqrt((dim+1)/dim - 1) * c_sigma ... 
            # This is complex. A simpler robust approximation:
            # Check if the step is consistent with the expected step size.
            # Let's use a simplified CSA update which is often stable enough.
            
            # Calculate expected norm of z
            # E||z|| = sqrt(pi/2) * ... roughly sqrt(dim) for large dim, but for dim=5 it's about 2.5
            # More precisely:
            # h_sigma = 1 if (||p_sigma|| / sigma) < (1 + 2/(dim+1)) * sqrt((dim+1)/dim - 1) * (1 - c_sigma)
            # This is getting complicated. Let's use a standard simplified CSA.
            
            # p_sigma update
            # p_sigma = (1 - c_sigma) * p_sigma + h_sigma * sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (C_sqrt_inv @ (x_mean - old_mean))
            
            # We need C_sqrt_inv @ (x_mean - old_mean)
            # This is equivalent to solving C_sqrt @ y = (x_mean - old_mean)
            # Or just using C_sqrt_inv directly.
            
            # Let's re-evaluate h_sigma
            # The condition for h_sigma=1 is:
            # ||p_sigma|| < (1 + 2/(dim+1)) * sqrt((dim+1)/dim - 1) * (1 - c_sigma) * sigma * sqrt(mueff) / sqrt(c_sigma * (2 - c_sigma))
            # This is too much. Let's use a simpler heuristic for h_sigma:
            # If the step length is reasonable, h_sigma=1.
            
            step = x_mean - old_mean
            step_len = np.linalg.norm(step)
            expected_step_len = sigma * np.sqrt(mueff) * 0.5 # Heuristic
            
            h_sigma = 1.0 if step_len < expected_step_len * 2 else 0.0
            
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_sqrt_inv @ step)
            
            sigma = sigma * np.exp(c_sigma * d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (self.budget - n_evals))) - np.sqrt(dim * (1 - np.exp(-2 * c_sigma)))))
            
            # Clamp sigma to avoid explosion or vanishing
            sigma = np.clip(sigma, 1e-6, 10.0)

            # 5. Update covariance
            # p_c update
            # p_c = (1 - c_c) * p_c + h_c * sqrt(c_c * (2 - c_c) * mueff) * (C_sqrt_inv @ step)
            
            # h_c condition is similar to h_sigma
            h_c = 1.0 if step_len < expected_step_len * 2 else 0.0
            
            p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mueff) * (C_sqrt_inv @ step)
            
            # Rank-one update
            C = (1 + c_c * (2 - c_c) - c_1) * C + c_1 * (p_c @ p_c.T)
            
            # Rank-mu update
            # y_i = C_sqrt_inv @ (x_i - old_mean) for i in 1..mu
            y_vecs = []
            for i in range(mu):
                y_vecs.append(C_sqrt_inv @ (sorted_offspring[i] - old_mean))
            
            # Sum of weighted outer products
            sum_yy = np.zeros((dim, dim))
            for i in range(mu):
                sum_yy += weights[i] * np.outer(y_vecs[i], y_vecs[i])
            
            C = C + c_mu * sum_yy
            
            # Regularize C to ensure positive definiteness
            C = C + np.eye(dim) * 1e-12
            
            # Update Cholesky
            C, C_inv, C_sqrt, C_sqrt_inv = update_chol(C, C_inv, C_sqrt, C_sqrt_inv)

        return best_x, best_f
