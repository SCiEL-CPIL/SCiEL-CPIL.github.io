import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        B = self.budget
        
        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(n)) # 7 for n=5
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Improved initialization: sample multiple points to find a better starting mean
        init_samples = 10
        init_points = np.random.uniform(-5, 5, (init_samples, n))
        best_init_idx = 0
        best_init_f = np.inf
        calls = 0
        for i in range(init_samples):
            f_val = func(init_points[i])
            calls += 1
            if f_val < best_init_f:
                best_init_f = f_val
                best_init_idx = i
        
        x = init_points[best_init_idx].copy()
        best_x = x.copy()
        best_f = best_init_f
        sigma = 2.0
        C = np.eye(n)
        C_inv_sqrt = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        def cholesky_inv_sqrt(M):
            # Returns matrix such that M = (L @ L.T) -> L_inv such that L_inv.T @ L_inv = M
            L = np.linalg.cholesky(M)
            L_inv = np.linalg.inv(L)
            return L_inv.T
            
        # We need to maintain C and its inverse square root for sampling
        # Efficient update: C_new = (1-c1-cmu)C + c1*p_c p_c.T + cmu sum wi (yi - mean)(yi - mean).T
        # We update C directly and recompute inverse sqrt periodically or use rank-one/two update.
        # For simplicity and robustness in a short script, we can recompute inv sqrt each step if n is small (5).
        
        while calls < B:
            # Sample
            ys = []
            for _ in range(lambda_):
                z = np.random.randn(n)
                y = x + sigma * (C_inv_sqrt @ z)
                y = np.clip(y, -5, 5)
                ys.append(y)
            
            # Evaluate
            fs = []
            for y in ys:
                f_y = func(y)
                calls += 1
                fs.append(f_y)
                if f_y < best_f:
                    best_f = f_y
                    best_x = y.copy()
                
                if calls >= B:
                    break
            
            if calls >= B:
                break
                
            fs = np.array(fs)
            ys = np.array(ys)
            
            # Sort
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]
            
            # Update mean
            y_mean = np.sum(weights[:, None] * ys[:mu], axis=0)
            
            # Update step size
            # p_sigma = (1-c_sigma) p_sigma + c_sigma * sqrt(mu_eff) * C_inv_sqrt @ (y_mean - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * (C_inv_sqrt @ (y_mean - x) / sigma)
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls + 1)))) / \
                   (np.sqrt(1 - (1 - c_sigma)**(2 * (calls + 1))) * np.sqrt(1 - np.exp(-2 * c_sigma * (calls + 1)))) if False else 1 # Simplified hsig
            # Correct hsig formula:
            norm_p_sigma = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) - 1 / (25 * n**2))
            hsig = 1 if norm_p_sigma / np.sqrt(1 - (1 - c_sigma)**(2 * (calls + 1))) < (1.4 + 2 / (n + 1)) * chi_n else 0
            
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            
            # Update correlation matrix
            p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_mean - x) / sigma
            
            # Update C
            # Delta C = -c1 * C + c1 * p_c p_c.T + cmu * sum wi (yi - y_mean)(yi - y_mean).T
            # To avoid numerical issues, we update C and recompute C_inv_sqrt
            C = (1 - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c)
            for i in range(mu):
                diff = ys[i] - y_mean
                C += c_mu * weights[i] * np.outer(diff, diff)
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update mean
            x = y_mean
            
            # Recompute C_inv_sqrt
            try:
                C_inv_sqrt = cholesky_inv_sqrt(C)
            except np.linalg.LinAlgError:
                # Fallback to identity if matrix becomes singular
                C = np.eye(n)
                C_inv_sqrt = np.eye(n)
                
        return best_x, best_f
