import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        if budget < 20:
            rng = np.random.default_rng(0)
            best_x = rng.uniform(-5, 5, n)
            best_f = func(best_x)
            for _ in range(budget - 1):
                x = rng.uniform(-5, 5, n)
                fx = func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            return best_x, best_f

        # CMA-ES Parameters
        lam = 10
        
        # Initialize mean (mu)
        mu = np.zeros(n)
        
        # Step size sigma
        sigma = 1.0
        
        # Covariance matrix C
        C = np.eye(n)
        
        # Weights for recombination
        mu_eff = lam // 2
        weights = np.zeros(mu_eff)
        log_w = np.log(np.arange(1, mu_eff + 1))
        weights[:] = np.exp(log_w - log_w[-1])
        weights /= np.sum(weights)
        
        # mu_eff = 1 / sum(w_i^2)
        mu_eff_val = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff_val + 2) / (n + mu_eff_val + 5)
        c_c = (4 + mu_eff_val / n) / (n + 4 + 2 * mu_eff_val / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff_val)
        c_mu = min(1 - c_1, 2 * (mu_eff_val - 2 + 2 / mu_eff_val) / ((n + 2)**2 + mu_eff_val))
        
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        # State variables
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        best_x = mu.copy()
        best_f = func(best_x)
        calls_made = 1
        
        rng = np.random.default_rng(42)
        
        while calls_made < budget:
            remaining = budget - calls_made
            current_lam = min(lam, remaining)
            is_last_step = (current_lam < lam)
            
            # Get Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-6 * np.eye(n)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(n)
                    C = np.eye(n)
            
            # Sample
            z_std = rng.normal(size=(current_lam, n))
            z = (L @ z_std.T).T
            pop = mu + sigma * z
            
            # Evaluate
            fvals = np.array([func(x) for x in pop])
            calls_made += current_lam
            
            # Update best
            idx = np.argsort(fvals)
            if fvals[idx[0]] < best_f:
                best_f = fvals[idx[0]]
                best_x = pop[idx[0]].copy()
                
            if is_last_step:
                break
            
            # Select top mu_eff individuals
            x_best = pop[idx[:mu_eff]]
            
            # New mean
            mu_new = np.sum(weights[:, None] * x_best, axis=0)
            
            # Update step size control
            try:
                L_inv = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                L_inv = np.eye(n)
                
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) / mu_eff_val) * (L_inv @ (mu_new - mu) / sigma)
            
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma = 1 if norm_p_sigma < (1.4 + 2/(n+1)) * chi_n else 0
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (norm_p_sigma / chi_n - 1))
            
            # Update covariance matrix
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) / mu_eff_val) * ((mu_new - mu) / sigma)
            
            y_vecs = (x_best - mu) / sigma
            
            C = (1 + c_c * (2 - c_c) * h_sigma - c_c - c_1 - c_mu) * C \
                + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_sigma * (2 - c_sigma) * C) \
                + c_mu * np.sum(weights[:, None, None] * y_vecs[:, :, None] * y_vecs[:, None, :], axis=0)
            
            C = (C + C.T) / 2.0
            mu = mu_new

        return best_x, best_f
