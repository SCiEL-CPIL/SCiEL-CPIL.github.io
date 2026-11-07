import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = 290.0
        n = self.dim
        
        best_x = np.zeros(n)
        best_f = np.inf
        total_evals = 0
        
        # Helper to run a CMA-ES instance
        def run_cmaes(x0, sigma0, max_evals):
            nonlocal total_evals, best_x, best_f
            nonlocal deadline, start_time
            
            # CMA-ES Parameters
            lambda_pop = 4 + int(3 * np.log(n))
            mu = lambda_pop // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
            
            x_mean = x0.copy()
            sigma = sigma0
            
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            C = np.eye(n)
            
            local_best_x = x_mean.copy()
            local_best_f = func(local_best_x)
            total_evals += 1
            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()
            
            last_improvement_eval = 0
            current_eval = 0
            
            while current_eval < max_evals:
                if time.time() - start_time > deadline:
                    return
                
                # Generate population
                offspring = []
                f_values = []
                
                for _ in range(lambda_pop):
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        C = np.eye(n)
                        L = np.eye(n)
                    
                    z = np.random.randn(n)
                    y = x_mean + sigma * (L @ z)
                    offspring.append(y)
                
                for y in offspring:
                    fy = func(y)
                    f_values.append(fy)
                    total_evals += 1
                    current_eval += 1
                    if fy < local_best_f:
                        local_best_f = fy
                        local_best_x = y.copy()
                        if fy < best_f:
                            best_f = fy
                            best_x = y.copy()
                            last_improvement_eval = current_eval
                
                if current_eval >= max_evals:
                    break
                    
                # Stagnation check
                if current_eval - last_improvement_eval > 40:
                    break
                
                # Sort
                indices = np.argsort(f_values)
                y_sorted = [offspring[i] for i in indices]
                
                # Update mean
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += weights[i] * y_sorted[i]
                x_mean_new = x_mean + sigma * y_w
                
                # Update sigma
                try:
                    L = np.linalg.cholesky(C)
                    L_inv_yw = np.linalg.solve(L, y_w)
                    norm_yw_C = np.linalg.norm(L_inv_yw)
                except np.linalg.LinAlgError:
                    norm_yw_C = np.linalg.norm(y_w)
                
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L_inv_yw / norm_yw_C if norm_yw_C > 1e-16 else 0)
                # Approximate h_sigma check (simplified)
                h_sigma = 1.0
                
                # Update covariance
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
                
                C = (1 + c_c * (2 - c_c) - c_1 - c_mu) * C \
                    + c_c * (p_c @ p_c.T + (1 - h_sigma) * c_c * (2 - c_c) * C) \
                    + c_mu * sum(weights[i] * np.outer(y_sorted[i], y_sorted[i]) for i in range(mu))
                
                C = (C + C.T) / 2
                try:
                    np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = C + 1e-6 * np.eye(n)
                    
                x_mean = x_mean_new
                sigma *= np.exp(c_sigma / d_sigma * (norm_yw_C / np.sqrt(mu_eff) - 1))
                x_mean = np.clip(x_mean, -5, 5)
                
            return local_best_x, local_best_f

        # Multi-start strategy
        max_restarts = 3
        for r in range(max_restarts):
            if total_evals >= self.budget:
                break
            if time.time() - start_time > deadline:
                break
            
            # Random start point
            x0 = np.random.uniform(-5, 5, n)
            # Initial sigma: larger for first restart, smaller for subsequent?
            # Or just standard.
            sigma0 = 2.0
            
            remaining = self.budget - total_evals
            # Allocate evals: First run gets more?
            # Let's give each run 1/3 of remaining budget?
            # Or just let it run until stagnation or budget.
            # Stagnation is the main trigger.
            run_cmaes(x0, sigma0, remaining)
            
        return best_x, best_f
