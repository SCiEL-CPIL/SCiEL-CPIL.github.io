import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        d = self.dim
        budget = self.budget
        
        # Parameters
        lambda_pop = 12
        mu = lambda_pop // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))
        
        x_mean = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        C_sqrt = np.eye(d)
        
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        gen = 0
        E_norm = np.sqrt(d) * (1 - 1.0/(4.0*d) + 1.0/(21.0*d**2))
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls = 1
        
        while calls + lambda_pop <= budget:
            if time.time() > deadline:
                break
            
            gen += 1
            
            # Sampling
            z = np.random.randn(lambda_pop, d)
            X = x_mean + sigma * (z @ C_sqrt.T)
            X = np.clip(X, -5, 5)
            
            F = np.array([func(x) for x in X])
            calls += lambda_pop
            
            idx_best = np.argmin(F)
            if F[idx_best] < best_f:
                best_f = F[idx_best]
                best_x = X[idx_best].copy()
                
            # Selection
            idx_sort = np.argsort(F)
            X_selected = X[idx_sort[:mu]]
            
            # Mean update
            x_mean_old = x_mean.copy()
            x_mean = x_mean + sigma * (weights @ X_selected)
            step = (x_mean - x_mean_old) / sigma
            
            # h_sigma calculation
            # ||p_sigma|| / sqrt(1 - (1-c_sigma)^(2*gen))
            # We need to accumulate the decay factor or just use the formula with gen.
            # The formula is:
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-c_sigma)^(2*gen)) < (1.4 + 2/(d+1)) * E_norm else 0
            
            # Note: p_sigma is updated AFTER this check? No, p_sigma is updated using the current step.
            # The check uses the p_sigma from the PREVIOUS iteration?
            # In Hansen's paper, p_sigma is updated first, then h_sigma is computed using the NEW p_sigma?
            # Actually, the condition is checked on the p_sigma vector *before* the update?
            # Let's check standard implementation.
            # Usually:
            # 1. Update x_mean
            # 2. Update p_sigma
            # 3. Compute h_sigma based on p_sigma (new)
            # 4. Update sigma
            # 5. Update p_c
            # 6. Compute delta_h based on p_c (new)
            # 7. Update C
            
            # Let's follow this order.
            
            p_sigma_new = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * step
            p_sigma = p_sigma_new
            
            # Compute h_sigma
            # The term sqrt(1 - (1-c_sigma)^(2*gen))
            # For large gen, this approaches 1.
            # For small gen, it's smaller.
            # We can compute it directly.
            decay_sigma = 1 - (1 - c_sigma) ** (2 * gen)
            if decay_sigma > 0:
                norm_p_sigma = np.linalg.norm(p_sigma) / np.sqrt(decay_sigma)
            else:
                norm_p_sigma = np.linalg.norm(p_sigma)
                
            h_sigma = 1.0 if norm_p_sigma < (1.4 + 2.0/(d+1)) * E_norm else 0.0
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / E_norm - 1))
            
            # Update p_c
            p_c_new = (1 - c_c) * p_c + h_sigma * c_c * step
            p_c = p_c_new
            
            # Compute delta_h
            decay_c = 1 - (1 - c_c) ** (2 * gen)
            if decay_c > 0:
                norm_p_c = np.linalg.norm(p_c) / np.sqrt(decay_c)
            else:
                norm_p_c = np.linalg.norm(p_c)
                
            delta_h = 1.0 if norm_p_c < (1.4 + 2.0/(d+1)) * E_norm else 0.0
            
            # Update C
            Y = (X_selected - x_mean_old) / sigma
            # Y shape (mu, d)
            
            C = (1 - c_1 - c_mu + mu_eff * c_mu) * C
            C += c_1 * (np.outer(p_c, p_c) + (1 - delta_h) * C)
            C += c_mu * (Y.T @ (Y * weights[:, None]))
            
            C = (C + C.T) / 2.0
            
            # Update C_sqrt
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C_sqrt = np.eye(d)
                C = np.eye(d)
                sigma *= 0.5

        return best_x, best_f
