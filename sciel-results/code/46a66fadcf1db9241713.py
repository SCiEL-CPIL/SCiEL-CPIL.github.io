import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        dim = self.dim
        total_budget = self.budget
        
        # State
        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_made = 0
        
        # Wrap func to track calls and handle exceptions
        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= total_budget:
                return best_f # Stop calling
            try:
                # Ensure x is 1D array
                x = np.asarray(x).flatten()
                f = func(x)
                calls_made += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                return f
            except Exception:
                return float('inf')
        
        # Phase 1: CMA-ES
        # Use 80% of budget for CMA-ES
        cma_budget = int(total_budget * 0.8)
        
        # CMA-ES Parameters
        pop_size = 10 # 4 + 3*ln(5) ~ 8, use 10 for robustness
        mean = np.zeros(dim)
        sigma = 1.0
        C = np.eye(dim)
        
        # CMA-ES State
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = (2 * (mu_eff - 2 + 2.0 / mu_eff)) / ((dim + 2) ** 2 + mu_eff)
        c_mu = min(c_mu, 2 * c_1)
        
        # Initial evaluation
        f0 = safe_func(mean)
        
        # CMA-ES Loop
        while calls_made < cma_budget:
            if time.time() > deadline:
                break
            
            # Generate population
            population = []
            fitness = []
            
            for _ in range(pop_size):
                if calls_made >= cma_budget:
                    break
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    L = np.eye(dim)
                
                z = np.random.randn(dim)
                x = mean + sigma * (L @ z)
                x = np.clip(x, -5, 5)
                
                f = safe_func(x)
                population.append(x)
                fitness.append(f)
            
            if len(population) < mu:
                break
                
            # Sort
            idx = np.argsort(fitness)
            population = [population[i] for i in idx]
            
            # Update Mean
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * population[i]
            
            # Update Sigma
            y = (new_mean - mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Update Covariance
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c)**2) * C) + c_mu * np.sum([np.outer(population[i] - mean, population[i] - mean) * weights[i] for i in range(mu)], axis=0)
            
            mean = new_mean
            
            # Check convergence
            if sigma < 1e-12:
                break
        
        # Phase 2: Nelder-Mead Polish
        # Use remaining budget
        remaining_budget = total_budget - calls_made
        if remaining_budget > 0 and best_f < float('inf'):
            # Define wrapper for scipy
            # scipy minimize expects func(x) -> float
            # We need to limit calls to remaining_budget
            nm_calls = 0
            def nm_func(x):
                nonlocal nm_calls
                if nm_calls >= remaining_budget:
                    return best_f
                nm_calls += 1
                try:
                    return func(np.asarray(x).flatten())
                except Exception:
                    return float('inf')
            
            # Run Nelder-Mead
            # maxiter is not the same as function calls, but we can limit it
            # A typical NM step uses dim+1 calls.
            # So maxiter ~ remaining_budget / (dim+1)
            maxiter = max(1, remaining_budget // (dim + 1))
            
            try:
                res = minimize(nm_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass
        
        return best_x, best_f
