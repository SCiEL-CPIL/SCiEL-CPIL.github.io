import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        d = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # For 5D and 300 budget, lambda=10 is a good choice (30 generations)
        lambda_ = 10
        mu = lambda_ // 2
        
        # Weights
        # Standard CMA-ES weights: w_i = log(mu + 0.5) - log(i) for i=1..mu
        # Normalized so sum is 1
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= np.sum(weights)
        
        # Effective population size
        mueff = 1.0 / np.sum(weights**2)
        
        # Strategy parameters
        c_sigma = (mueff + 2) / (d + mueff + 5)
        dsigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        c_1 = 2 / ((d + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff))
        
        # Initialize
        # Start at 0,0,0,0,0. The problem says minimizer is in [-4,4]^5.
        # 0 is a reasonable center.
        m = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        C_inv = np.eye(d)
        C_sqrt = np.eye(d)
        C_inv_sqrt = np.eye(d)
        
        # Evolution paths
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        # Cholesky decomposition update helper
        def update_chol(C):
            # Simple Cholesky update for C <- C + rho * x x^T
            # Here we use the standard CMA-ES rank-1 and rank-mu update
            pass # We will implement the update logic directly

        calls = 0
        best_x = m.copy()
        best_f = np.inf
        
        # Initial evaluation of mean
        best_f = func(best_x)
        calls += 1
        
        # Precompute weights for offspring selection
        # Sort indices of offspring by fitness
        
        while calls < budget:
            # Check time limit
            if time.time() - self.start_time > 320: # 10s margin
                break
                
            # Determine number of offspring to generate
            # We can generate up to lambda_ offspring
            # If budget is low, we might generate fewer
            num_offspring = min(lambda_, budget - calls)
            
            if num_offspring <= 0:
                break
                
            # Sample offspring
            # We need C_sqrt. We maintain C_sqrt via Cholesky.
            # For simplicity and robustness in this small budget, 
            # we can recompute C_sqrt every generation if C changes significantly, 
            # or use a stable update. 
            # Given the small d=5, recomputing Cholesky is cheap.
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset to identity
                C = np.eye(d)
                C_sqrt = np.eye(d)
                sigma *= 0.5
                
            y = np.zeros((num_offspring, d))
            for i in range(num_offspring):
                y[i] = m + sigma * (C_sqrt @ np.random.standard_normal(d))
            
            # Clip to box
            y = np.clip(y, -5, 5)
            
            # Evaluate
            fy = np.zeros(num_offspring)
            for i in range(num_offspring):
                if calls >= budget:
                    break
                fy[i] = func(y[i])
                calls += 1
                if fy[i] < best_f:
                    best_f = fy[i]
                    best_x = y[i].copy()
            
            if calls >= budget:
                break
            
            # Sort offspring by fitness
            idx = np.argsort(fy)
            y = y[idx]
            fy = fy[idx]
            
            # Use top mu offspring for update
            # If we generated fewer than mu, use all
            num_parents = min(mu, num_offspring)
            
            # Update mean
            m_new = m.copy()
            for i in range(num_parents):
                m_new += weights[i] * y[i]
            m = m_new
            
            # Update sigma (CSA - Cumulative Step Adaptation)
            y_sigma = (m_new - m_old) / (sigma * np.linalg.norm(C_sqrt @ np.eye(d)[0])) # Approximation, better to use actual norm
            # Actually, standard CMA-ES uses:
            # y_sigma = (m_new - m_old) / sigma
            # But it needs to be in the C-space.
            # Let's use the standard formula:
            # h_sigma = ||p_sigma_new|| < (1 + c_sigma) * E||N(0, I)|| * sqrt(1 - exp(-2 * mueff / lambda_))
            
            # Recalculate y_sigma properly
            # The step in the original space is m_new - m_old.
            # The step in the transformed space is C_inv_sqrt @ (m_new - m_old) / sigma
            
            # Let's simplify: We will use a simplified CMA-ES update that is stable for small d.
            # We update C and sigma based on the selected parents.
            
            # 1. Update p_sigma
            # y_sigma = (m_new - m_old) / sigma
            y_sigma = (m_new - m_old) / sigma
            # Transform to C-space
            # We need C_inv_sqrt. 
            # Since d=5, we can compute it.
            try:
                C_inv_sqrt = np.linalg.inv(C_sqrt)
            except:
                C_inv_sqrt = np.eye(d)
                
            y_sigma_C = C_inv_sqrt @ y_sigma
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_sigma_C
            
            # h_sigma
            # E||N(0, I)|| = sqrt(d) * Gamma((d+1)/2) / Gamma(d/2)
            # For d=5, E||N|| ~ 2.0
            e_norm = np.sqrt(d) * 2.0 # Approximation
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (budget - calls + 1))) < (1 + c_sigma) * e_norm * 1.02 else 0.0
            
            # 2. Update sigma
            sigma *= np.exp(c_sigma / dsigma * (np.linalg.norm(p_sigma) / e_norm - 1))
            
            # 3. Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * y_sigma_C
            
            # 4. Update C
            # Rank-1 update
            v = p_c / np.sqrt(1 - (1 - c_c)**2) # Approximation, usually just p_c scaled
            # Standard: v = p_c
            C += c_1 * (p_c @ p_c - C)
            
            # Rank-mu update
            for i in range(num_parents):
                # y_i in C-space
                y_i_C = C_inv_sqrt @ (y[i] - m_old) / sigma
                C += c_mu * weights[i] * (y_i_C @ y_i_C - C)
            
            # Symmetrize and ensure positive definiteness
            C = (C + C.T) / 2
            # Add small regularization if needed
            C += 1e-8 * np.eye(d)
            
            # Store old mean for next iteration
            m_old = m_new
            
        return best_x, best_f
