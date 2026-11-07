import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        D = self.dim
        B = self.budget
        
        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(D))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        # Initial state
        x_mean = np.zeros(D)
        sigma = 1.0
        C = np.eye(D)
        C_sqrt = np.linalg.cholesky(C)
        
        # Adaptation parameters
        cs = (mu_eff + 2) / (D + mu_eff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (D + 1)) - 1) + cs
        cc = (4 + mu_eff / D) / (D + 4 + 2 * mu_eff / D)
        c1 = 2 / ((D + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((D + 2)**2 + mu_eff))
        
        # History
        p_sigma = np.zeros(D)
        p_c = np.zeros(D)
        
        best_x = x_mean.copy()
        best_f = np.inf
        calls = 0
        
        def evaluate(x):
            nonlocal best_x, best_f, calls
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        # Evaluate initial mean
        evaluate(x_mean)
        
        x_mean_old = x_mean.copy()
        
        while calls + lam <= B:
            # Generate population
            pop = np.zeros((lam, D))
            f_pop = np.zeros(lam)
            
            for i in range(lam):
                y = np.random.randn(D)
                x = x_mean + sigma * (C_sqrt @ y)
                x = np.clip(x, -5.0, 5.0)
                pop[i] = x
                f_pop[i] = evaluate(x)
                
            # Sort
            idx = np.argsort(f_pop)
            pop = pop[idx]
            f_pop = f_pop[idx]
            
            # Update mean
            x_mean_new = np.zeros(D)
            for i in range(mu):
                x_mean_new += w[i] * pop[i]
                
            # Update p_sigma
            # y_mean = (x_mean_new - x_mean_old) / sigma
            # p_sigma = (1-cs)*p_sigma + sqrt(cs*(2-cs)*mu_eff) * y_mean
            # Note: We use the unclipped direction for adaptation to maintain scale, 
            # but clipping might distort it. For simplicity and robustness, we use the raw difference.
            # However, if clipping occurs, the actual step is different. 
            # Standard CMA-ES assumes no boundaries or handles them via reflection.
            # Given the tight bounds, simple clipping is used. This might slightly bias adaptation 
            # but is acceptable for a small budget.
            
            y_mean = (x_mean_new - x_mean_old) / sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * y_mean
            
            # Update sigma
            # sigma = sigma * exp(cs/ds * (||p_sigma|| / chiN - 1))
            chi_n = np.sqrt(D) * (1 - 1/(4*D) + 1/(21*D**2))
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma = sigma * np.exp(cs / ds * (norm_p_sigma / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)
            
            # Update p_c
            # p_c = (1-cc)*p_c + cc * h * sqrt(mu_eff) * y_mean
            # h = 1 if ||p_sigma||/sqrt(1-(1-cs)^{2/(gen+1)}) < (1.4+2/(D+1)) * chi_n
            # We approximate the history length or just use a simple threshold.
            # For short runs, we can approximate h=1 or use a simple check.
            # Let's use a simple check: if norm_p_sigma < 1.4 * chi_n, h=1 else 0.
            # This is a heuristic but works well for short budgets.
            h = 1.0 if norm_p_sigma < 1.4 * chi_n else 0.0
            p_c = (1 - cc) * p_c + cc * h * np.sqrt(mu_eff) * y_mean
            
            # Update Covariance
            # C = (1 - c1 - cmu + c1*mu_eff*h) * C + c1 * (p_c p_c^T + (1-h)*cc*(1-cc)*C)
            #    + cmu * sum(w_i * y_i y_i^T)
            
            # Calculate y_i for selected individuals
            # y_i = (pop[i] - x_mean_old) / sigma
            # Note: We use x_mean_old as the center for the covariance update in standard CMA-ES?
            # Actually, the standard update uses the mean of the *current* generation as the center for the rank-mu update?
            # No, the rank-mu update is based on the vectors from the *previous* mean.
            # y_i = (x_i - x_mean_old) / sigma
            
            rank_mu = np.zeros((D, D))
            for i in range(mu):
                y_i = (pop[i] - x_mean_old) / sigma
                rank_mu += w[i] * np.outer(y_i, y_i)
                
            C = (1 - c1 - cmu + c1 * mu_eff * h) * C
            if h == 0:
                C += c1 * cc * (1 - cc) * C
            else:
                C += c1 * np.outer(p_c, p_c)
            C += cmu * rank_mu
            
            # Symmetrize and ensure positive definite
            C = (C + C.T) / 2
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If Cholesky fails, add a small diagonal
                C += 1e-10 * np.eye(D)
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback to identity if completely broken
                    C = np.eye(D)
                    C_sqrt = np.eye(D)
            
            x_mean_old = x_mean_new
            x_mean = x_mean_new
            
        return best_x, best_f
