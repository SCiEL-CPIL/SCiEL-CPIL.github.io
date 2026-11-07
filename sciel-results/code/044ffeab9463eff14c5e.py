"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Pure CMA-ES optimizer for Sharp Ridge (f13)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        n = self.dim
        # Population size: (4 + ln(n)) * n
        # For n=5: (4 + 1.609) * 5 = 28.04 -> 29
        # Standard rule is lambda = 4 + 3 ln(n) for some variants, or (4+ln n)n
        # Let's use the standard CMA-ES recommendation: lambda = 4 + 3 ln(n) is for (1+lambda) ES often.
        # For CMA-ES, lambda ~ (4 + ln(n)) * n is common.
        # n=5 -> 28. Let's use 28.
        lambda_pop = int(np.ceil((4 + np.log(n)) * n))
        
        # Initial sigma: 2.5 is reasonable for [-5,5] box (std dev covers ~50% of range)
        # But with rotation/shift, starting at 0 with sigma 2.0-2.5 is safe.
        sigma = 2.0
        
        # Mean
        m = np.zeros(n)
        
        # Covariance matrix
        C = np.eye(n)
        
        # Step size
        # sigma is the step size
        
        # Parameters
        mu = lambda_pop // 2  # Number of selected individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Covariance matrix adaptation
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        # Initial state
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        def time_ok():
            return time.time() < deadline
            
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Generate initial population
        # We need to manage the call budget carefully.
        # Each generation uses lambda_pop calls.
        
        while calls < self.budget and time_ok():
            # Generate offspring
            offspring = []
            # We will generate lambda_pop individuals
            # To ensure we don't exceed budget mid-generation, we check before each call
            
            for k in range(lambda_pop):
                if calls >= self.budget or not time_ok():
                    break
                    
                # Sample from N(m, sigma^2 C)
                # Use Cholesky decomposition or SVD for stability
                # For small n, np.linalg.cholesky is fast
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback if C becomes singular
                    L = np.linalg.cholesky(C + 1e-12 * np.eye(n))
                    
                z = np.random.randn(n)
                y = L @ z
                x = m + sigma * y
                
                # Clip to box [-5, 5]
                x = np.clip(x, -5, 5)
                
                f = func(x)
                calls += 1
                update_best(x, f)
                
                offspring.append((x, f, y))
            
            if len(offspring) == 0:
                break
            
            # Sort offspring by fitness
            offspring.sort(key=lambda x: x[1])
            
            # Select best mu
            selected = offspring[:mu]
            
            # Compute new mean
            m_new = sum(weights[i] * selected[i][0] for i in range(mu))
            
            # Update step size control vector ps
            inv_sigma = 1.0 / sigma
            # C^-1/2 @ (m_new - m)
            # Compute C^-1/2
            # Since C = L @ L.T, C^-1/2 = (L^-1).T @ L^-1 ? No.
            # C^-1/2 is the matrix such that C^-1/2 @ C @ C^-1/2 = I
            # Using eigenvalue decomposition is more stable for this
            w, v = np.linalg.eigh(C)
            w = np.clip(w, 1e-16, None) # Ensure positive
            C_inv_sqrt = v @ np.diag(1.0 / np.sqrt(w)) @ v.T
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (m_new - m) * inv_sigma
            
            # Check hsig
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls // lambda_pop + 1))) < (1.4 + 2 / (n + 1)) * chi_n
            
            # Update evolution path pc
            if hsig:
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (m_new - m) * inv_sigma
            else:
                pc = (1 - c_c) * pc
            
            # Update covariance matrix
            h1 = (1 - c_1)
            h1mu = 1 - c_1 - c_mu
            
            # Rank-one update
            # (pc + hsig - 1) @ (pc + hsig - 1).T
            rank_one = (pc + (hsig - 1) * np.sqrt(c_c * (2 - c_c) * mu_eff) * (m_new - m) * inv_sigma)
            # Actually, the standard formula is:
            # C = (1 - c_1 - c_mu + c_mu * theta) C + c_1 * (pc @ pc.T + (hsig - 1) * c_c * (2 - c_c) * (m_new - m) @ (m_new - m).T * inv_sigma^2)
            # Wait, the term in rank one is related to pc.
            # Let's stick to the standard CMA-ES update formula from Hansen's paper.
            
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (hsig - 1) * c_c * (2 - c_c) * y_best @ y_best.T)
            # where y_best = (m_new - m) / sigma
            # And the rank-mu update:
            # + c_mu * sum(weights[i] * y_i @ y_i.T)
            
            y_best = (m_new - m) * inv_sigma
            rank_one_term = pc @ pc.T + (hsig - 1) * c_c * (2 - c_c) * (y_best @ y_best.T)
            
            rank_mu_term = np.zeros((n, n))
            for i in range(mu):
                y_i = selected[i][2] # The y from sampling
                rank_mu_term += weights[i] * (y_i @ y_i.T)
                
            C = (1 - c_1 - c_mu) * C + c_1 * rank_one_term + c_mu * rank_mu_term
            
            # Symmetrize C
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update mean
            m = m_new
            
            # Check convergence
            # If sigma is too small or step size is too small
            if sigma < 1e-15:
                break
            if np.max(np.linalg.norm(C, axis=1)) < 1e-15:
                break
            # If progress is negligible
            if calls > 50 and (best_f - min(f for _, f, _ in offspring)) < 1e-12:
                 # Optional: restart logic could go here, but for simplicity and budget constraints, 
                 # we might just continue or break if stuck. 
                 # Given the "Sharp Ridge", CMA-ES usually handles it well without restarts 
                 # if initialized well. 
                 pass

        return best_x, best_f

# EVOLVE-BLOCK-END
