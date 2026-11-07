import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        
        f_global_best = np.inf
        x_global_best = None
        
        # We will do up to 5 restarts of 100 calls each? 
        # Or just one long run?
        # Let's try 5 restarts of 100 calls.
        # This allows us to try different basins.
        
        n_restarts = 5
        calls_per_restart = budget // n_restarts
        
        for _ in range(n_restarts):
            # Initialize
            x = rng.uniform(-5, 5, d)
            f_best = func(x)
            if f_best < f_global_best:
                f_global_best = f_best
                x_global_best = x.copy()
                
            lam = 4 + int(3 * np.log(d))
            if lam < 4: lam = 4
            if lam > 20: lam = 20
            
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1 / np.sum(w**2)
            
            c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
            c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
            chi_n = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
            
            sigma = 1.0
            C = np.eye(d)
            p_sigma = np.zeros(d)
            p_c = np.zeros(d)
            
            calls_in_restart = 1
            last_improvement_call = 0
            
            while calls_in_restart < calls_per_restart:
                z = rng.standard_normal((lam, d))
                L = np.linalg.cholesky(C)
                y = z @ L.T
                pop = x[None, :] + sigma * y
                
                f_vals = np.array([func(xi) for xi in pop])
                calls_in_restart += lam
                
                idx_best = np.argmin(f_vals)
                if f_vals[idx_best] < f_best:
                    f_best = f_vals[idx_best]
                    if f_best < f_global_best:
                        f_global_best = f_best
                        x_global_best = pop[idx_best].copy()
                    last_improvement_call = calls_in_restart
                    
                order = np.argsort(f_vals)
                y_sorted = pop[order]
                y_w = np.sum(w[:, None] * y_sorted[:mu], axis=0)
                
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * (L.T @ y_w)
                
                if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_in_restart // lam + 1))) < (1.4 + 2 / (d + 1)) * chi_n:
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                    
                x = x + sigma * y_w
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
                
                C = (1 - c_sigma) * C + c_sigma * (d_sigma * np.outer(p_sigma, p_sigma) + 
                     np.sum(w[:, None, None] * (y_sorted[:mu] - y_w[None, :])[:, :, None] * 
                            (y_sorted[:mu] - y_w[None, :])[:, None, :], axis=0))
                C = (C + C.T) / 2
                x = np.clip(x, -5, 5)
                
                # Stall detection
                if calls_in_restart - last_improvement_call > 50:
                    break
                    
                if sigma < 1e-8:
                    break
                    
        return x_global_best, f_global_best
