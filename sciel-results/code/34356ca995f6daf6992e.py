import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0 # Safety margin
        
        dim = self.dim
        total_budget = self.budget
        
        # Global best
        x_best = None
        f_best = np.inf
        
        # CMA-ES Parameters
        # For dim=5, N=10 is standard. 
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
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        rng = np.random.default_rng(42)
        
        used_calls = 0
        restart_count = 0
        
        # Strategy:
        # 1. Use a smaller population (N=4) for faster initial exploration to improve early AOCC.
        # 2. Switch to standard CMA-ES (N=10) for exploitation once some progress is made.
        # 3. Use restarts based on stagnation.
        
        # Phase 1: Fast exploration with small population
        N_small = 4
        mu_small = N_small // 2
        weights_small = np.log(mu_small + 0.5) - np.log(np.arange(1, mu_small + 1))
        weights_small /= np.sum(weights_small)
        mu_eff_small = 1.0 / np.sum(weights_small ** 2)
        
        c_sigma_small = (mu_eff_small + 2) / (dim + mu_eff_small + 5)
        d_sigma_small = 1 + 2 * max(0, np.sqrt((mu_eff_small - 1) / (dim + 1)) - 1) + c_sigma_small
        c_c_small = (4 + mu_eff_small / dim) / (dim + 4 + 2 * mu_eff_small / dim)
        c_1_small = 2 / ((dim + 1.3) ** 2 + mu_eff_small)
        c_mu_small = min(1 - c_1_small, 2 * (mu_eff_small - 2 + 1 / mu_eff_small) / ((dim + 2) ** 2 + mu_eff_small))
        
        # We will run multiple short CMA-ES instances
        # First few with N_small, then with N_full
        
        max_restarts = 10
        min_improve_threshold = 1e-8
        
        for restart in range(max_restarts):
            if used_calls >= total_budget or time.time() > deadline:
                break
                
            # Choose starting point
            if restart == 0:
                x_start = np.zeros(dim)
            else:
                # Mix of random and perturbation of best
                if f_best < np.inf and rng.random() < 0.5:
                    x_start = x_best + rng.normal(0, 0.5, dim)
                    x_start = np.clip(x_start, -5, 5)
                else:
                    x_start = rng.uniform(-5, 5, dim)
            
            # Decide population size
            if restart < 3 and used_calls < 50:
                N_curr = N_small
                mu_curr = mu_small
                weights_curr = weights_small
                mu_eff_curr = mu_eff_small
                c_sigma_curr = c_sigma_small
                d_sigma_curr = d_sigma_small
                c_c_curr = c_c_small
                c_1_curr = c_1_small
                c_mu_curr = c_mu_small
            else:
                N_curr = N
                mu_curr = mu
                weights_curr = weights
                mu_eff_curr = mu_eff
                c_sigma_curr = c_sigma
                d_sigma_curr = d_sigma
                c_c_curr = c_c
                c_1_curr = c_1
                c_mu_curr = c_mu
            
            # Initialize CMA-ES state
            mean = x_start.copy()
            sigma = 2.0 if restart == 0 else 1.0
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            min_sigma = 1e-12
            stagnation_limit = 5 # Generations without improvement
            no_improve_count = 0
            run_calls = 0
            best_run_f = np.inf
            best_run_x = mean.copy()
            
            # Evaluate start
            if used_calls >= total_budget:
                break
            f_mean = func(mean)
            used_calls += 1
            run_calls += 1
            if f_mean < f_best:
                f_best = f_mean
                x_best = mean.copy()
            best_run_f = f_mean
            best_run_x = mean.copy()
            
            # Main loop for this run
            while used_calls + N_curr <= total_budget and time.time() < deadline:
                if sigma < min_sigma:
                    break
                
                # Sample
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                
                B = rng.standard_normal((N_curr, dim)) @ L.T
                Y = mean + sigma * B
                Y = np.clip(Y, -5, 5)
                
                # Evaluate
                F = np.empty(N_curr)
                for i in range(N_curr):
                    if used_calls >= total_budget or time.time() > deadline:
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
                
                if used_calls + N_curr > total_budget or time.time() > deadline:
                    break
                    
                # Sort
                I = np.argsort(F)
                Y_sorted = Y[I]
                F_sorted = F[I]
                
                # Stagnation check
                if F_sorted[0] < best_run_f - 1e-12:
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                if no_improve_count > stagnation_limit:
                    break
                
                # Update mean
                mean_new = np.sum(weights_curr[:, np.newaxis] * Y_sorted[:mu_curr], axis=0)
                
                # Update sigma
                y_sigma = (mean_new - mean) / (sigma * np.sqrt(mu_eff_curr))
                p_sigma = (1 - c_sigma_curr) * p_sigma + np.sqrt(c_sigma_curr * (2 - c_sigma_curr) * mu_eff_curr) * y_sigma
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma_curr) ** (2 * (run_calls + 1))) / np.sqrt(mu_eff_curr) < (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff_curr + 1) / (mu_eff_curr - 1)))
                
                sigma *= np.exp(c_sigma_curr / d_sigma_curr * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff_curr) - 1))
                
                # Update p_c
                p_c = (1 - c_c_curr) * p_c + h_sigma * np.sqrt(c_c_curr * (2 - c_c_curr) * mu_eff_curr) * y_sigma
                h_c = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c_curr) ** (2 * (run_calls + 1))) / np.sqrt(mu_eff_curr) < (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff_curr + 1) / (mu_eff_curr - 1)))
                
                # Update C
                rank_one = np.outer(p_c, p_c)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu_curr):
                    z_i = (Y_sorted[i] - mean) / sigma
                    rank_mu += weights_curr[i] * np.outer(z_i, z_i)
                    
                C = (1 + c_1_curr * h_c - c_1_curr - c_mu_curr) * C + c_1_curr * rank_one + c_mu_curr * rank_mu
                mean = mean_new
                
                # Convergence check
                if np.all(np.abs(mean - x_start) < 1e-6) or sigma < 1e-10:
                    break
            
            # End of run
            restart_count += 1

        return x_best, f_best
