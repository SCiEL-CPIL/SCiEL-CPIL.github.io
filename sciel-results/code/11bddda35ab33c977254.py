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
        
        # Multi-start CMA-ES strategy
        # We will run multiple short CMA-ES instances.
        # Each instance gets a portion of the budget.
        # If an instance stagnates, we restart.
        
        dim = self.dim
        budget = self.budget
        
        # Initial global best
        self.best_x = np.random.uniform(-5, 5, dim)
        self.best_f = func(self.best_x)
        self.calls += 1
        
        if self.calls >= budget or time.time() > deadline:
            return self.best_x, self.best_f

        # Parameters for CMA-ES
        # We use a smaller population for faster restarts
        # Standard CMA-ES with lam=4 is fast but prone to getting stuck.
        # Let's try lam=4 but with aggressive restarts.
        
        # Budget allocation:
        # We will run CMA-ES in "phases".
        # Phase 1: Exploration with small steps.
        # Phase 2: Exploitation.
        
        # Actually, let's use a simple restart policy:
        # Run CMA-ES for 'max_evals_per_restart' evaluations.
        # If the best value doesn't improve by a certain factor, restart.
        
        max_evals_per_restart = 50
        min_improvement = 1e-6
        
        current_budget_used = self.calls
        
        while current_budget_used < budget and time.time() < deadline:
            # Start a new CMA-ES instance
            # Initialize mean near the current best or random?
            # To escape local minima, initialize randomly or near best with large sigma.
            # Let's initialize near the best point but with a large sigma to allow escape.
            # Or, just random if we haven't found anything good.
            
            if self.best_f > 0: # Arbitrary threshold, or just always restart near best with large sigma
                mean = self.best_x + np.random.uniform(-1, 1, dim)
                sigma = 2.0
            else:
                mean = np.random.uniform(-5, 5, dim)
                sigma = 2.0
            
            # Clip mean to box
            mean = np.clip(mean, -5, 5)
            
            # CMA-ES State
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
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
            
            mean_prev = mean.copy()
            
            # Run CMA-ES for this restart
            evals_in_restart = 0
            best_f_in_restart = self.best_f
            last_best_f = self.best_f
            stagnation_counter = 0
            
            while evals_in_restart < max_evals_per_restart:
                if current_budget_used >= budget or time.time() > deadline:
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
                    if current_budget_used >= budget or time.time() > deadline:
                        break
                    x_i = np.clip(pop[i], -5, 5)
                    f_i = func(x_i)
                    f_vals.append((f_i, x_i))
                    current_budget_used += 1
                    self.calls += 1
                    
                    if f_i < self.best_f:
                        self.best_f = f_i
                        self.best_x = x_i
                        best_f_in_restart = f_i
                
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
                denom = np.sqrt(1 - (1 - c_sigma)**(2 * (evals_in_restart + 1)))
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
                evals_in_restart += 1
                
                # Check for stagnation
                if best_f_in_restart < last_best_f - 1e-8:
                    last_best_f = best_f_in_restart
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                    
                # If stagnation is high, break early to restart
                if stagnation_counter > 15:
                    break

        return self.best_x, self.best_f
