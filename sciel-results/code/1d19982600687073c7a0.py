import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Time guard
        start_time = time.time()
        deadline = start_time + 280 # Leave 20s margin for 300s limit
        
        rng = np.random.default_rng(42)
        
        # CMA-ES Parameters
        lam = 10
        mu = lam // 2
        
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        w = np.concatenate([w, np.zeros(lam - mu)])
        
        mueff = 1.0 / np.sum(w**2)
        
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = (2 - mueff / (dim + 2) + c_c) / ((dim + 1.3) ** 2 + mueff + c_c)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        
        # Initialization
        x_mean = np.zeros(dim)
        sigma = 3.0 # Initial step size, covering a good portion of [-5,5]
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = x_mean.copy()
        best_f = np.inf
        
        evals_used = 0
        
        # Helper to decompose C
        def get_cholesky(C):
            try:
                L = np.linalg.cholesky(C)
                return L
            except np.linalg.LinAlgError:
                # Add regularization if not PD
                C += 1e-6 * np.eye(dim)
                L = np.linalg.cholesky(C)
                return L

        L = get_cholesky(C)
        
        # Initial evaluation
        f_mean = func(x_mean)
        evals_used += 1
        best_f = f_mean
        best_x = x_mean.copy()
        
        # CMA-ES Loop
        while evals_used < budget:
            if time.time() > deadline:
                break
                
            if budget - evals_used < lam:
                break

            # Generate offspring
            ys = []
            xs = []
            fs = []
            
            for i in range(lam):
                if evals_used >= budget or time.time() > deadline:
                    break
                    
                z = rng.standard_normal(dim)
                y = L @ z
                x_new = x_mean + sigma * y
                x_new = np.clip(x_new, -5, 5)
                
                f_val = func(x_new)
                evals_used += 1
                
                ys.append(y)
                xs.append(x_new)
                fs.append(f_val)
                
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_new.copy()
                    
            if len(xs) < lam:
                break
                
            ys = np.array(ys)
            xs = np.array(xs)
            fs = np.array(fs)
            
            idx = np.argsort(fs)
            xs = xs[idx]
            ys = ys[idx]
            fs = fs[idx]
            
            x_mean_old = x_mean.copy()
            x_mean = np.sum(w[:mu, None] * xs[:mu], axis=0)
            
            # Update evolution paths
            # Compute C_inv_sqrt
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ (x_mean - x_mean_old) / sigma)
            
            norm_ps = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim**2))
            
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c)) * (C_inv_sqrt @ (x_mean - x_mean_old) / sigma)
            
            # Update covariance
            sum_w_y = np.zeros((dim, dim))
            for i in range(mu):
                sum_w_y += w[i] * np.outer(ys[i], ys[i])
                
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c)**2) * C) + c_mu * sum_w_y
            
            C = (C + C.T) / 2
            C += 1e-12 * np.eye(dim)
            
            L = get_cholesky(C)
            
        # Local Refinement using (1+1)-ES with CMA-ES geometry
        remaining = budget - evals_used
        if remaining > 0 and time.time() < deadline:
            x_local = best_x.copy()
            f_local = best_f
            
            # Use a small sigma for local search, scaled by the CMA-ES sigma if it's reasonable
            # If CMA-ES sigma is very small, use a small fixed value. If large, use a fraction.
            sigma_local = max(sigma * 0.5, 0.1)
            
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                
                # Generate step using the CMA-ES covariance structure (L)
                # This aligns the local search with the landscape anisotropy
                z = rng.standard_normal(dim)
                y = L @ z
                x_new = x_local + sigma_local * y
                x_new = np.clip(x_new, -5, 5)
                
                f_new = func(x_new)
                evals_used += 1
                
                if f_new < f_local:
                    f_local = f_new
                    x_local = x_new.copy()
                    best_f = f_local
                    best_x = x_local.copy()
                    sigma_local *= 1.05 # Slight increase on success
                else:
                    sigma_local *= 0.95  # Decrease on failure
                    
                if sigma_local < 1e-6:
                    sigma_local = 1e-6

        return best_x, best_f
