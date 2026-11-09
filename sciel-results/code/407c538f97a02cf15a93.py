import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        n = self.dim
        total_budget = self.budget
        calls_made = 0
        best_x = np.zeros(n)
        best_f = np.inf

        def do_call(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= total_budget:
                return np.inf
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)
        
        # CMA-ES Parameters
        # Population size: 5 * (n + 3) is a common robust choice, but we can start smaller to conserve budget
        # For n=5, 5*(5+3) = 40. This allows for good exploration.
        # Let's use a population size that fits well within budget.
        # Budget is 500. 
        # If pop_size=20, we can do 25 generations.
        # If pop_size=10, we can do 50 generations.
        # For Rastrigin, larger population helps escape local minima.
        # Let's try pop_size = 2 * (n + 3) + 1 = 11? Or maybe just a fixed 20-30.
        # Standard CMA-ES often uses lambda = 4 + floor(3 ln n). For n=5, 4 + 3*1.6 = 8.8 -> 8 or 9.
        # However, for multimodal functions like Rastrigin, a larger population is beneficial.
        # Let's use pop_size = 20.
        
        pop_size = 20
        sigma = 2.0 # Initial step size, covering a large portion of the box [-5,5]
        x_mean = np.zeros(n)
        
        # Initial covariance matrix (identity)
        C = np.eye(n)
        
        # Precompute weights
        mu = pop_size // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mu_eff = 1 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff - 2 + 2 / mu_eff) / ((n + 2)**2 + mu_eff)))
        
        # Damping for covariance update
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        # Memory for path updates
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Cache for evaluated points in this generation to avoid duplicates (optional but good)
        # Given the continuous nature, exact duplicates are rare, but we can check if we want to save budget.
        # For now, we assume distinct points.
        
        generation = 0
        while calls_made + pop_size <= total_budget and time_ok():
            generation += 1
            
            # Generate offspring
            offspring = []
            offspring_values = []
            
            # Update covariance matrix before generating new points? 
            # Standard CMA-ES: Generate, Evaluate, Sort, Update.
            
            for i in range(pop_size):
                # Sample from multivariate normal
                z = rng.standard_normal(n)
                # Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C becomes non-positive definite, reset or repair
                    C = np.eye(n)
                    L = np.eye(n)
                
                x_i = x_mean + sigma * (L @ z)
                x_i = np.clip(x_i, -5.0, 5.0)
                
                f_i = do_call(x_i)
                offspring.append(x_i)
                offspring_values.append(f_i)
                
                if calls_made >= total_budget or not time_ok():
                    break
            
            if len(offspring) < pop_size:
                break
                
            # Sort offspring by fitness
            sorted_indices = np.argsort(offspring_values)
            sorted_offspring = [offspring[i] for i in sorted_indices]
            sorted_values = [offspring_values[i] for i in sorted_indices]
            
            # Update mean
            x_mean = sum(weights[i] * sorted_offspring[i] for i in range(mu))
            
            # Update sigma (path adaptation)
            # p_sigma = (1 - c_sigma) * p_sigma + c_sigma * sqrt(mu_eff) / chi_n * (C^(-1/2) @ (x_new_mean - x_old_mean)) / sigma
            # We need C^(-1/2). We have L such that C = L L^T. So C^(-1/2) = (L^T)^(-1)
            try:
                L_inv = np.linalg.inv(L.T)
            except np.linalg.LinAlgError:
                L_inv = np.eye(n)
                
            x_diff = (x_mean - x_mean_old) / sigma if 'x_mean_old' in locals() else np.zeros(n)
            # Actually, we should store x_mean_old before updating x_mean
            # Let's restructure slightly to store old mean
            
            # Correction: Store old mean before update
            # I will rewrite the loop body to be more careful with state updates.
            
            # Let's restart the logic inside the loop for clarity and correctness.
            
            break # Break out to rewrite properly
            
        # Rewriting the main loop properly
        
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        generation = 0
        while calls_made + pop_size <= total_budget and time_ok():
            generation += 1
            x_mean_old = x_mean.copy()
            
            # Generate population
            offspring = []
            offspring_values = []
            
            # Get current Cholesky factor
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                L = np.eye(n)
            
            for i in range(pop_size):
                z = rng.standard_normal(n)
                x_i = x_mean + sigma * (L @ z)
                x_i = np.clip(x_i, -5.0, 5.0)
                
                f_i = do_call(x_i)
                offspring.append(x_i)
                offspring_values.append(f_i)
                
                if calls_made >= total_budget or not time_ok():
                    break
            
            if len(offspring) < pop_size:
                break
                
            # Sort
            sorted_indices = np.argsort(offspring_values)
            sorted_offspring = [offspring[i] for i in sorted_indices]
            sorted_values = [offspring_values[i] for i in sorted_indices]
            
            # Update Mean
            x_mean = sum(weights[i] * sorted_offspring[i] for i in range(mu))
            
            # Update Covariance and Sigma
            # Path for sigma
            try:
                L_inv = np.linalg.inv(L.T)
            except np.linalg.LinAlgError:
                L_inv = np.eye(n)
                
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) / chi_n * (L_inv @ (x_mean - x_mean_old) / sigma)
            
            # Update sigma
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            # Cap sigma to prevent explosion or collapse
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Path for covariance
            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) / chi_n * (x_mean - x_mean_old) / sigma
            
            # Rank-one and rank-mu updates
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c + (1 - (1 - c_c)^2) * C) + c_mu * sum(weights[i] * (y_i @ y_i))
            # where y_i = (x_i - x_mean_old) / sigma
            
            y_vecs = [(x - x_mean_old) / sigma for x in sorted_offspring[:mu]]
            
            rank_mu = sum(weights[i] * np.outer(y_vecs[i], y_vecs[i]) for i in range(mu))
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c)**2) * C) + c_mu * rank_mu
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Regularization: if C becomes ill-conditioned, add small diagonal
            eigvals = np.linalg.eigvalsh(C)
            if np.any(eigvals < 1e-10):
                C += 1e-10 * np.eye(n)

        return best_x, best_f
