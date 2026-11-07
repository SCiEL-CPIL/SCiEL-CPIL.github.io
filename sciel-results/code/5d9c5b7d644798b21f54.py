import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        t_start = time.time()
        max_time = 290.0
        
        # CMA-ES parameters
        lam = 10
        if lam < 4: lam = 4
        mu = lam // 2
        
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1 / np.sum(w**2)
        
        # Strategy parameters
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        rng = np.random.default_rng(42)
        
        evals = 0
        best_x = np.zeros(n)
        best_f = np.inf
        
        # Initial evaluation
        x_init = np.zeros(n)
        f_init = func(x_init)
        evals += 1
        best_x = x_init.copy()
        best_f = f_init
        
        # Restart logic
        chunk_size = 100
        stagnation_limit = 50
        
        while evals < budget:
            if time.time() - t_start > max_time:
                break
                
            # Initialize CMA-ES state for this run
            m = best_x.copy()
            sigma = 1.0 
            pc = np.zeros(n)
            ps = np.zeros(n)
            C = np.eye(n)
            C_sqrt = np.linalg.cholesky(C)
            inv_C_sqrt = np.linalg.inv(C_sqrt)
            
            run_evals = 0
            last_improvement_eval = 0
            
            # Run CMA-ES for this chunk
            while run_evals < chunk_size and evals < budget:
                if time.time() - t_start > max_time:
                    break
                
                # Generate offspring
                X = np.zeros((lam, n))
                F = np.zeros(lam)
                
                for i in range(lam):
                    if evals >= budget: break
                    z = rng.standard_normal(n)
                    y = C_sqrt @ z
                    x = m + sigma * y
                    X[i] = x
                    F[i] = func(x)
                    evals += 1
                    run_evals += 1
                    
                    if F[i] < best_f:
                        best_f = F[i]
                        best_x = x.copy()
                        last_improvement_eval = run_evals

                if run_evals < lam:
                    break
                    
                # Sort
                idx = np.argsort(F)
                X = X[idx]
                F = F[idx]
                
                # Update Mean
                y_weighted = np.zeros(n)
                for i in range(mu):
                    y_weighted += w[i] * (X[i] - m) / sigma
                
                # Update Step-size
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (inv_C_sqrt @ y_weighted)
                chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
                h_sigma = 1 if (np.linalg.norm(ps) / chi_n) < (1.4 + 2/(n+1)) else 0
                if h_sigma == 1:
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                
                # Update Covariance
                pc = (1 - c_c) * pc + c_c * y_weighted
                delta_sigma = c_sigma * (2 - c_sigma) * mueff
                C_new = (1 - c_1 - c_mu) * C 
                C_new += c_1 * (np.outer(pc, pc) + (1 - delta_sigma) * C)
                
                z_mat = (X[:mu] - m) / sigma
                for i in range(mu):
                    C_new += c_mu * w[i] * np.outer(z_mat[i], z_mat[i])
                    
                C_new = (C_new + C_new.T) / 2
                try:
                    C_sqrt = np.linalg.cholesky(C_new + 1e-9 * np.eye(n))
                    inv_C_sqrt = np.linalg.inv(C_sqrt)
                    C = C_new
                except np.linalg.LinAlgError:
                    C_sqrt = np.linalg.cholesky(C + 1e-8 * np.eye(n))
                    inv_C_sqrt = np.linalg.inv(C_sqrt)
                    break
                
                m = m + sigma * y_weighted
                
                # Check for stagnation within this run
                if run_evals - last_improvement_eval > stagnation_limit:
                    break
                    
                if sigma < 1e-8:
                    break

        return best_x, best_f
