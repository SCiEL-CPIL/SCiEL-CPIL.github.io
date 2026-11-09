import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(n))
        
        x_mean = np.zeros(n)
        sigma = 2.0
        
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        C = np.eye(n)
        
        def get_C_sqrt_inv(C):
            evals, evecs = np.linalg.eigh(C)
            evals = np.clip(evals, 1e-12, None)
            sqrt_evals = np.sqrt(evals)
            C_sqrt = evecs @ np.diag(sqrt_evals) @ evecs.T
            C_inv_sqrt = evecs @ np.diag(1.0 / sqrt_evals) @ evecs.T
            return C_sqrt, C_inv_sqrt

        C_sqrt, C_inv_sqrt = get_C_sqrt_inv(C)
        
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        best_x = x_mean.copy()
        best_f = func(x_mean)
        calls = 1
        
        while budget - calls >= lambda_:
            ys = []
            fs = []
            for _ in range(lambda_):
                z = np.random.randn(n)
                x = x_mean + sigma * C_sqrt @ z
                f = func(x)
                calls += 1
                ys.append(z)
                fs.append(f)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    
            indices = np.argsort(fs)
            ys = np.array(ys)[indices]
            fs = np.array(fs)[indices]
            
            y_selected = ys[:mu]
            
            x_old = x_mean
            x_mean = x_mean + sigma * np.sum(weights[:, None] * y_selected, axis=0)
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ ((x_mean - x_old) / sigma)
            
            # Check for h_sigma
            # The denominator is sqrt(1 - (1-c_sigma)^(2 * generations))
            # generations is approximately calls / lambda_
            # But simpler: use the standard formula with current generation count
            # gen_count = calls / lambda_ is not integer.
            # Let's use the approximation from the CMA-ES reference:
            # h_sigma = ||p_sigma|| / sqrt(n) < 1.4 + 2/(n+1)
            # The normalization factor in the reference is sqrt(1 - (1-c_sigma)^(2 * t))
            # where t is the number of generations.
            # For simplicity and robustness, I will use the simpler condition without the time-dependent normalization, 
            # or approximate t.
            t = calls / lambda_
            norm_factor = np.sqrt(1 - (1 - c_sigma)**(2 * t)) if t > 0 else 1.0
            if norm_factor == 0: norm_factor = 1e-8
            h_sigma = (np.linalg.norm(p_sigma) / norm_factor) / np.sqrt(n) < 1.4 + 2 / (n + 1)
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean - x_old) / sigma
            
            # Update C
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.sum(weights[:, None] * (y_selected * y_selected.T), axis=0)
            
            C = (1 + c_c * (2 - c_c) - c_1 - c_mu * delta_h) * C \
                + c_1 * (rank_one + delta_h * np.outer(p_c, p_c)) \
                + c_mu * (rank_mu - C)
            
            C = (C + C.T) / 2
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(n) - 1))
            
            C_sqrt, C_inv_sqrt = get_C_sqrt_inv(C)
            
            if sigma < 1e-8:
                break

        # Handle remaining budget
        rem = budget - calls
        if rem > 0 and sigma > 1e-8:
            for _ in range(rem):
                z = np.random.randn(n)
                x = x_mean + sigma * C_sqrt @ z
                f = func(x)
                calls += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

        return best_x, best_f
