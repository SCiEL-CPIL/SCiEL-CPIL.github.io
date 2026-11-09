import numpy as np
import time
import scipy.stats.qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        
        # 1. Global Scan
        n_scan = 40
        sampler = scipy.stats.qmc.Sobol(d=self.dim, scramble=True, seed=42)
        points = sampler.random(n_scan)
        points = points * 10 - 5
        
        best_x = None
        best_f = np.inf
        candidates = []
        
        for x in points:
            if time.time() > deadline - 10.0: break
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            candidates.append((fx, x.copy()))
            
        if best_x is None:
             x0 = rng.uniform(-5, 5, self.dim)
             return x0, func(x0)

        # 2. CMA-ES based local refinement
        remaining = self.budget - len(candidates)
        if remaining <= 0:
            return best_x, best_f

        candidates.sort(key=lambda t: t[0])
        
        def cma_es(start_x, start_f, n_evals, max_sigma=5.0, min_sigma=1e-6):
            """CMA-ES implementation for f21 optimization"""
            dim = self.dim
            # Population size
            lam = max(4, int(4 + 3 * np.log(dim)))
            
            # CMA-ES state
            mean = start_x.copy()
            sigma = max(0.5, min(2.0, max_sigma / 3))
            C = np.eye(dim)
            C_inv = np.eye(dim)
            
            # Strategy parameters
            alpha_mu = 0.5
            mu = lam // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (weights[0] - 1 + alpha_mu) / ((dim + 2)**2 + mu_eff))
            
            # Evolution paths
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            best_x_local = start_x.copy()
            best_f_local = start_f
            evals_used = 0
            stagnation_counter = 0
            
            for gen in range(n_evals // lam + 1):
                if time.time() > deadline - 5.0:
                    break
                if evals_used >= n_evals:
                    break
                    
                # Generate population
                offspring = []
                for i in range(lam):
                    if time.time() > deadline - 5.0 or evals_used >= n_evals:
                        break
                    # Sample
                    z = rng.standard_normal(dim)
                    y = mean + sigma * (C @ z)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    evals_used += 1
                    offspring.append((fy, y))
                    
                if len(offspring) == 0:
                    break
                    
                # Sort by fitness
                offspring.sort(key=lambda t: t[0])
                
                # Update best
                if offspring[0][0] < best_f_local:
                    best_f_local = offspring[0][0]
                    best_x_local = offspring[0][1].copy()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                
                # Check for stagnation - restart if stuck
                if stagnation_counter > 3:
                    # Try a local jump
                    jump = rng.standard_normal(dim) * sigma * 2
                    new_x = np.clip(mean + jump, -5, 5)
                    new_f = func(new_x)
                    evals_used += 1
                    if new_f < best_f_local:
                        best_f_local = new_f
                        best_x_local = new_x.copy()
                    mean = best_x_local.copy()
                    sigma = max(sigma * 0.5, min_sigma)
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
                    stagnation_counter = 0
                    continue
                
                # Update mean
                y_w = np.zeros(dim)
                for i in range(mu):
                    y_w += weights[i] * offspring[i][1]
                new_mean = y_w
                
                # Update evolution path p_sigma
                C_sqrt_inv = np.linalg.cholesky(np.eye(dim) - C)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv @ (new_mean - mean)) / sigma
                h_sigma = 1 if np.sqrt(np.sum(p_sigma**2) / (1 - (1 - c_sigma)**(2 * (evals_used // lam + 1)))) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (dim + 1)) * np.sqrt(mu_eff + 1 - 1) else 0
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_mean - mean) / sigma
                
                # Update covariance
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                rank_one = np.outer(p_c, p_c)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    y_i = (offspring[i][1] - mean) / sigma
                    rank_mu += weights[i] * np.outer(y_i, y_i)
                
                C = (1 + c_1 * (2 - c_1 - delta_h) - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
                C_inv = np.linalg.inv(C)
                
                mean = new_mean
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                sigma = max(min_sigma, min(sigma, max_sigma))
                
                # Ensure C is positive definite
                try:
                    np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = 0.5 * C + 0.5 * np.eye(dim)
                    C_inv = np.linalg.inv(C)
            
            return best_x_local, best_f_local

        # Allocate budget: 60% to best, 20% to second, 20% to third if available
        n_top = min(3, len(candidates))
        budgets = []
        if n_top >= 1:
            budgets.append(int(remaining * 0.6))
        if n_top >= 2:
            budgets.append(int(remaining * 0.2))
        if n_top >= 3:
            budgets.append(remaining - sum(budgets))
        
        # Run CMA-ES on top candidates
        for i in range(n_top):
            if time.time() > deadline - 10.0:
                break
            x_start, f_start = candidates[i]
            x_res, f_res = cma_es(x_start, f_start, budgets[i])
            if f_res < best_f:
                best_x = x_res
                best_f = f_res

        # 3. Final Polish with very small steps
        polish_budget = 30
        if time.time() < deadline - 5.0:
            polish_sigma = 0.02
            for _ in range(polish_budget):
                if time.time() > deadline - 3.0:
                    break
                y = np.clip(best_x + polish_sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                if fy < best_f:
                    best_x = y
                    best_f = fy
                polish_sigma *= 0.92

        return best_x, best_f
