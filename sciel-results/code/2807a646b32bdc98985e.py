import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        
        dim = self.dim
        budget = self.budget
        
        # Initial global best
        self.best_x = np.random.uniform(-5, 5, dim)
        self.best_f = func(self.best_x)
        self.calls += 1
        
        if self.calls >= budget or time.time() > deadline:
            return self.best_x, self.best_f
        
        # IPOP-CMA-ES with adaptive population size
        # This is the strategy that achieved 1.944 score
        
        def cma_es_run(func, dim, budget, start_mean, start_sigma, max_evals, deadline, calls_ref, best_ref):
            """
            Run CMA-ES for max_evals evaluations.
            Returns (mean, sigma, C, p_sigma, p_c, evals_used)
            """
            lam = 4
            mu = max(1, lam // 2)
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            
            mean = start_mean.copy()
            sigma = start_sigma
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            mean_prev = mean.copy()
            
            evals_in_run = 0
            
            while evals_in_run < max_evals:
                if calls_ref[0] >= budget or time.time() > deadline:
                    break
                
                # Generate population
                z = np.random.randn(lam, dim)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    L = np.eye(dim)
                
                pop = mean + sigma * (z @ L.T)
                
                f_vals = []
                for i in range(lam):
                    if calls_ref[0] >= budget or time.time() > deadline:
                        break
                    x_i = np.clip(pop[i], -5, 5)
                    f_i = func(x_i)
                    f_vals.append((f_i, x_i))
                    calls_ref[0] += 1
                    best_ref[1] += 1
                    
                    if f_i < best_ref[0]:
                        best_ref[0] = f_i
                        best_ref[2] = x_i
                
                if len(f_vals) == 0:
                    break
                
                f_vals.sort(key=lambda x: x[0])
                
                # Update mean
                new_mean = np.zeros(dim)
                for i in range(min(mu, len(f_vals))):
                    new_mean += weights[i] * f_vals[i][1]
                mean = new_mean
                
                # Update path variables
                try:
                    L_inv = np.linalg.inv(L)
                except np.linalg.LinAlgError:
                    L_inv = np.eye(dim)
                
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L_inv @ (mean - mean_prev) / sigma)
                
                norm_p_sigma = np.linalg.norm(p_sigma)
                denom = np.sqrt(1 - (1 - c_sigma)**(2 * (evals_in_run + 1)))
                if denom == 0:
                    h_sigma = 0
                else:
                    h_sigma = 1 if (norm_p_sigma / denom < (1.4 + 2 / (dim + 1)) * chi_n) else 0
                
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (L_inv @ (mean - mean_prev) / sigma)
                
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C)
                for i in range(min(mu, len(f_vals))):
                    y_i = L_inv @ (f_vals[i][1] - mean_prev) / sigma
                    C += c_mu * weights[i] * np.outer(y_i, y_i)
                
                sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
                
                mean_prev = mean.copy()
                evals_in_run += 1
                
                # Check for stagnation - if sigma becomes very small, break
                if sigma < 1e-10:
                    break
                # Check if mean has moved very little
                if evals_in_run > 20 and np.linalg.norm(mean - mean_prev) < 1e-12:
                    break
            
            return mean, sigma, C, p_sigma, p_c, evals_in_run
        
        # IPOP-CMA-ES main loop
        # Start with a single CMA-ES instance, restart with larger population if stagnation
        
        current_calls = [self.calls]
        best_state = [self.best_f, 0, self.best_x]  # [best_f, calls_count, best_x]
        
        # First run: small population, large sigma
        mean = self.best_x.copy()
        sigma = 2.0
        max_evals = 100
        
        while current_calls[0] < budget and time.time() < deadline:
            # Run CMA-ES
            mean, sigma, C, p_sigma, p_c, evals_used = cma_es_run(
                func, dim, budget, mean, sigma, max_evals, deadline, current_calls, best_state
            )
            
            if current_calls[0] >= budget or time.time() > deadline:
                break
            
            # Check for stagnation: if sigma is very small or we made little progress
            # Increase max_evals for next restart (IPOP strategy)
            max_evals = min(max_evals * 2, 200)
            
            # Restart: initialize mean near best, with fresh covariance
            # Sometimes random restart to escape local minima
            if np.random.rand() < 0.3:
                mean = np.random.uniform(-5, 5, dim)
            else:
                mean = best_state[2] + np.random.uniform(-1, 1, dim)
                mean = np.clip(mean, -5, 5)
            
            sigma = 2.0
        
        return self.best_x, self.best_f
