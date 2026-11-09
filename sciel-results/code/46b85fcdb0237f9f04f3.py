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
        deadline = start_time + 290  # Leave 10s margin for 300s limit
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Random sampling to get initial landscape understanding
        n_random = 50
        best_x = None
        best_f = np.inf
        evals_used = 0
        
        # Evaluate center first
        x0 = np.zeros(dim)
        f0 = func(x0)
        evals_used += 1
        best_x = x0.copy()
        best_f = f0
        
        # Random sampling
        for _ in range(n_random):
            if evals_used >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, size=dim)
            f = func(x)
            evals_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Phase 2: CMA-ES with smaller population for efficiency
        lam = 6
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        w = np.concatenate([w, np.zeros(lam - mu)])
        
        mueff = 1.0 / np.sum(w**2)
        
        # Strategy parameters
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = (2 - mueff / (dim + 2) + c_c) / ((dim + 1.3) ** 2 + mueff + c_c)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        
        # Start from the best random point
        x_mean = best_x.copy()
        sigma = 1.5  # Smaller initial step for faster convergence
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Helper to decompose C
        def get_cholesky(C):
            C_reg = C + 1e-10 * np.eye(dim)
            try:
                L = np.linalg.cholesky(C_reg)
                return L
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C_reg)
                eigvals = np.clip(eigvals, 1e-12, None)
                L = eigvecs @ np.diag(np.sqrt(eigvals))
                return L

        L = get_cholesky(C)
        
        # Convergence detection
        no_improvement_count = 0
        max_no_improvement = 12
        
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
            
            if fs[0] < best_f - 1e-10:
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if no_improvement_count >= max_no_improvement:
                break
            
            x_mean_old = x_mean.copy()
            x_mean = np.sum(w[:mu, None] * xs[:mu], axis=0)
            
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ (x_mean - x_mean_old) / sigma)
            
            norm_ps = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim**2))
            
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c)) * (C_inv_sqrt @ (x_mean - x_mean_old) / sigma)
            
            sum_w_y = np.zeros((dim, dim))
            for i in range(mu):
                sum_w_y += w[i] * np.outer(ys[i], ys[i])
                
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c)**2) * C) + c_mu * sum_w_y
            
            C = (C + C.T) / 2
            C += 1e-12 * np.eye(dim)
            
            L = get_cholesky(C)
            
        # Phase 3: Local refinement using (1+1)-ES
        remaining = budget - evals_used
        if remaining > 0 and time.time() < deadline:
            x_local = best_x.copy()
            f_local = best_f
            sigma_local = max(sigma, 0.05)
            
            stagnant = 0
            max_stagnant = 25
            
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                
                z = rng.standard_normal(dim)
                x_new = x_local + sigma_local * z
                x_new = np.clip(x_new, -5, 5)
                
                f_new = func(x_new)
                evals_used += 1
                
                if f_new < f_local:
                    f_local = f_new
                    x_local = x_new.copy()
                    best_f = f_local
                    best_x = x_local.copy()
                    stagnant = 0
                    sigma_local *= 1.03
                else:
                    stagnant += 1
                    sigma_local *= 0.97
                
                if stagnant >= max_stagnant:
                    sigma_local = max(sigma_local * 1.5, 0.3)
                    stagnant = 0
                
                sigma_local = np.clip(sigma_local, 1e-6, 3.0)
                    
        return best_x, best_f
