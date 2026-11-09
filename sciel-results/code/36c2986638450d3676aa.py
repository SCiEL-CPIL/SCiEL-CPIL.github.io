import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        start_time = time.time()
        deadline = start_time + 290
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        evals_used = 0
        
        def eval_point(x):
            nonlocal best_x, best_f, evals_used
            x = np.clip(x, -5, 5)
            f = func(x)
            evals_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Global Initialization
        # Sample points to find a good starting basin
        init_candidates = []
        for _ in range(10):
            if evals_used >= budget: break
            x = rng.uniform(-5, 5, dim)
            f = eval_point(x)
            init_candidates.append((f, x))
        
        init_candidates.sort(key=lambda k: k[0])
        x_mean = init_candidates[0][1].copy()
        best_f = init_candidates[0][0]
        
        # Phase 2: CMA-ES
        # Parameters
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
        
        sigma = 2.0 # Larger initial step size to cover the domain
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Precompute L
        L = np.eye(dim)
        
        cma_budget = budget - evals_used
        cma_evals = 0
        
        while cma_evals < cma_budget and evals_used < budget:
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
                f_val = eval_point(x_new)
                
                ys.append(y)
                xs.append(x_new)
                fs.append(f_val)
                
            if len(xs) < lam:
                break
                
            cma_evals += lam
            
            ys = np.array(ys)
            xs = np.array(xs)
            fs = np.array(fs)
            
            # Sort by fitness
            idx = np.argsort(fs)
            xs = xs[idx]
            ys = ys[idx]
            fs = fs[idx]
            
            # Update mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum(w[:mu, None] * xs[:mu], axis=0)
            
            # Update C for next step's sampling
            # Calculate C_inv_sqrt
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ (x_mean - x_mean_old) / sigma)
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c)) * (C_inv_sqrt @ (x_mean - x_mean_old) / sigma)
            
            # Update sigma
            norm_ps = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim**2))
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            
            # Update covariance
            sum_w_y = np.zeros((dim, dim))
            for i in range(mu):
                sum_w_y += w[i] * np.outer(ys[i], ys[i])
                
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c)**2) * C) + c_mu * sum_w_y
            C = (C + C.T) / 2
            C += 1e-12 * np.eye(dim)
            
            # Update L
            L = np.linalg.cholesky(C)
            
            # Stop if converged
            if sigma < 1e-4:
                break

        return best_x, best_f
