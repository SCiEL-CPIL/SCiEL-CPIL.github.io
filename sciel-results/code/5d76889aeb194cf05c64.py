import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42) # Fixed seed for reproducibility in dev, but task says "hidden seeded shift", so rng inside optimizer should be fixed or random? 
        # "func(x) ... Instances apply a hidden seeded shift". This implies the *function* is fixed per instance. 
        # The optimizer's internal randomness can be fixed to ensure deterministic behavior for the same function instance.
        
        # CMA-ES Parameters
        lam = 9
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1 / np.sum(w**2)
        
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        chi_n = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
        
        # State
        m = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        calls = 0
        best_x = m.copy()
        best_f = func(m)
        calls += 1
        
        # Pre-compute Cholesky if possible
        try:
            L = np.linalg.cholesky(C)
        except:
            L = np.eye(d)
            
        while calls < budget:
            # Generate Population
            pop = []
            for _ in range(lam):
                # Check budget
                if calls >= budget:
                    break
                
                z = rng.standard_normal(d)
                try:
                    y = L @ z
                except:
                    y = z # Fallback if C becomes singular
                x = m + sigma * y
                x = np.clip(x, -5, 5)
                f = func(x)
                pop.append((f, x))
                calls += 1
                
            if not pop:
                break
                
            # Sort
            pop.sort(key=lambda t: t[0])
            
            # Update Best
            if pop[0][0] < best_f:
                best_f = pop[0][0]
                best_x = pop[0][1]
                
            # Update Mean
            m_old = m.copy()
            m = np.sum([w[i] * pop[i][1] for i in range(mu)], axis=0)
            
            # Update Sigma
            y_bar = (m - m_old) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * y_bar
            # h_sigma condition
            # Approximation for the norm check
            norm_p_sigma = np.linalg.norm(p_sigma)
            # The denominator is sqrt(1 - (1-c_sigma)^(2 * t)) where t is generation count
            # We can approximate or just use a fixed threshold if generation count is low
            # For simplicity and robustness, we can use the standard threshold
            h_sigma = norm_p_sigma / chi_n < (1.4 + 2 / (d + 1))
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            
            # Update Covariance
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_bar
            
            # Rank 1 + Rank mu update
            # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1-h_sigma) c_c (2-c_c) C) + c_mu sum w_i y_i y_i^T
            C_new = (1 - c_1 - c_mu) * C
            C_new += c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C)
            
            for i in range(mu):
                y_i = (pop[i][1] - m_old) / sigma
                C_new += c_mu * w[i] * np.outer(y_i, y_i)
                
            C = C_new
            
            # Update Cholesky
            try:
                L = np.linalg.cholesky(C)
            except:
                # If C is not positive definite, regularize
                C = C + 1e-8 * np.eye(d)
                try:
                    L = np.linalg.cholesky(C)
                except:
                    L = np.eye(d)
                    
        # Fine Polish Phase
        # If we have budget left and sigma is small, do a local search
        # Actually, the loop stops when budget is reached.
        # The CMA-ES itself polishes as sigma decreases.
        
        return best_x, best_f
