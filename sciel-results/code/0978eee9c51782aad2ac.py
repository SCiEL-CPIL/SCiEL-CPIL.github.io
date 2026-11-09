import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        
        # Time guard
        start_time = time.time()
        time_limit = 290.0
        
        # Global best
        x_best = np.zeros(dim)
        f_best = np.inf
        
        # CMA-ES Parameters
        N = 10
        mu = N // 2
        
        # Pre-compute weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        rng = np.random.default_rng(42) # Use a fixed seed for reproducibility in testing, but random in prod? 
        # The task says "hidden seeded shift", so the function is different each time.
        # We should probably use a random seed for the optimizer to avoid bias towards specific instances?
        # Or just use a default rng.
        
        used_calls = 0
        restart_count = 0
        
        # Strategy: Restart CMA-ES
        init_sigmas = [2.5, 2.0, 1.5, 1.0, 0.5, 0.2, 0.1]
        
        while used_calls < total_budget and time.time() - start_time < time_limit and restart_count < 10:
            x_start = rng.uniform(-5, 5, dim)
            
            if restart_count < len(init_sigmas):
                sigma_init = init_sigmas[restart_count]
            else:
                sigma_init = rng.uniform(0.05, 0.2)
            
            mean = x_start.copy()
            sigma = sigma_init
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            min_sigma = 1e-12
            stagnation_limit = 20
            
            run_calls = 0
            best_run_x = x_start.copy()
            best_run_f = np.inf
            
            if used_calls + 1 > total_budget:
                break
                
            f_start = func(x_start)
            used_calls += 1
            run_calls += 1
            
            if f_start < f_best:
                f_best = f_start
                x_best = x_start.copy()
            best_run_f = f_start
            best_run_x = x_start.copy()
            
            no_improve_count = 0
            gen_count = 0
            
            while used_calls + N <= total_budget and time.time() - start_time < time_limit:
                if sigma < min_sigma:
                    break
                    
                gen_count += 1
                
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    L = eigvecs @ np.diag(np.sqrt(eigvals))
                    
                B = rng.standard_normal((N, dim)) @ L.T
                Y = mean + sigma * B
                Y = np.clip(Y, -5, 5)
                
                F = np.empty(N)
                for i in range(N):
                    if used_calls >= total_budget or time.time() - start_time > time_limit:
                        break
                    F[i] = func(Y[i])
                    used_calls += 1
                    run_calls += 1
                    
                    if F[i] < best_run_f:
                        best_run_f = F[i]
                        best_run_x = Y[i].copy()
                        if F[i] < f_best:
                            f_best = F[i]
                            x_best = Y[i].copy()
                
                if used_calls + N > total_budget or time.time() - start_time > time_limit:
                    break
                    
                I = np.argsort(F)
                Y_sorted = Y[I]
                F_sorted = F[I]
                
                if F_sorted[0] < best_run_f - 1e-12:
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                if no_improve_count > stagnation_limit:
                    break
                
                mean_new = np.sum(weights[:, np.newaxis] * Y_sorted[:mu], axis=0)
                
                y_sigma = (mean_new - mean) / (sigma * np.sqrt(mu_eff))
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
                
                # FIX: Use gen_count for the decay factor
                # The term is sqrt(1 - (1-c)^{2t}) where t is the iteration count
                decay_sigma = np.sqrt(1 - (1 - c_sigma) ** (2 * gen_count))
                chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
                h_sigma = (np.linalg.norm(p_sigma) / decay_sigma < (1.4 + 2/(dim + 1)) * chi_n)
                
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma
                
                decay_c = np.sqrt(1 - (1 - c_c) ** (2 * gen_count))
                h_c = (np.linalg.norm(p_c) / decay_c < (1.4 + 2/(dim + 1)) * chi_n)
                
                rank_one = np.outer(p_c, p_c)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    z_i = (Y_sorted[i] - mean) / sigma
                    rank_mu += weights[i] * np.outer(z_i, z_i)
                    
                C = (1 + c_1 * h_c - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
                mean = mean_new
                
                if np.linalg.norm(mean - mean_new) < 1e-6 * sigma:
                    break
                    
            restart_count += 1

        return x_best, f_best
