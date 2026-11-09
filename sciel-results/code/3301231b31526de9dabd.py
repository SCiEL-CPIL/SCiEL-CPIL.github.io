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
        deadline = start_time + 290.0
        
        rng = np.random.default_rng(42)
        
        # CMA-ES Parameters
        # Using a larger population size (15) than the inspiration (10) to better handle
        # potential discontinuities or "steps" in the landscape and avoid premature convergence.
        lam = 15
        mu = lam // 2
        
        # Weights for recombination
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        w = np.concatenate([w, np.zeros(lam - mu)])
        
        mueff = 1.0 / np.sum(w**2)
        
        # Adaptation parameters
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = (2 - mueff / (dim + 2) + c_c) / ((dim + 1.3) ** 2 + mueff + c_c)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        
        # Initial state
        # Start at center, with a step size that covers a good portion of the domain
        x_mean = np.zeros(dim)
        sigma = 2.0
        
        C = np.eye(dim)
        L = np.linalg.cholesky(C)
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = x_mean.copy()
        best_f = np.inf
        
        evals_used = 0
        
        # Initial evaluation
        f_mean = func(x_mean)
        evals_used += 1
        if f_mean < best_f:
            best_f = f_mean
            best_x = x_mean.copy()
            
        # Phase 1: CMA-ES
        # Run CMA-ES for ~80% of the budget to allow for full convergence of the covariance matrix
        cma_budget = int(budget * 0.8)
        
        while evals_used < cma_budget:
            if time.time() > deadline:
                break
                
            if budget - evals_used < lam:
                break

            # Generate offspring
            zs = rng.standard_normal((lam, dim))
            ys = zs @ L.T
            xs = x_mean + sigma * ys
            xs = np.clip(xs, -5, 5)
            
            # Evaluate
            fs = np.zeros(lam)
            for i in range(lam):
                if evals_used >= budget or time.time() > deadline:
                    break
                f_val = func(xs[i])
                evals_used += 1
                fs[i] = f_val
                if f_val < best_f:
                    best_f = f_val
                    best_x = xs[i].copy()
                    
            # If we broke out early due to budget/time, we can't do a full generation update
            if evals_used >= budget or time.time() > deadline:
                break

            idx = np.argsort(fs)
            xs = xs[idx]
            ys = ys[idx]
            fs = fs[idx]
            
            x_mean_old = x_mean.copy()
            x_mean = np.sum(w[:mu, None] * xs[:mu], axis=0)
            
            # Update p_sigma
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ (x_mean - x_mean_old) / sigma)
            
            norm_ps = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim**2))
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            
            # Update p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c)) * (C_inv_sqrt @ (x_mean - x_mean_old) / sigma)
            
            y_bar = (x_mean - x_mean_old) / sigma
            
            # Vectorized sum_w_y
            sum_w_y = np.sum(w[:mu, None, None] * ys[:mu, :, None] * ys[:mu, None, :], axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c)**2) * C) + c_mu * sum_w_y
            
            C = (C + C.T) / 2
            C += 1e-12 * np.eye(dim)
            
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If Cholesky fails, increase regularization and retry
                C += 1e-6 * np.eye(dim)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback: reset covariance to identity if severely broken
                    C = np.eye(dim)
                    L = np.linalg.cholesky(C)
                    sigma = 1.0

        # Phase 2: Local refinement
        # Use remaining budget for fine-tuning with a simple (1+1)-ES or Nelder-Mead style search
        remaining = budget - evals_used
        if remaining > 0 and time.time() < deadline and best_x is not None:
            x_local = best_x.copy()
            f_local = best_f
            
            # Start with a small step size relative to current sigma
            sigma_local = max(sigma * 0.1, 1e-3)
            
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
                    sigma_local *= 1.1  # Expand slightly on success
                else:
                    sigma_local *= 0.8  # Shrink on failure
                    
                if sigma_local < 1e-8:
                    sigma_local = 1e-8

        return best_x, best_f
