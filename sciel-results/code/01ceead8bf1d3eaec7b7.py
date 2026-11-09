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
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        best_x = np.zeros(n)
        best_f = float('inf')
        calls_used = 0

        def check_deadline():
            return time.time() >= deadline

        def do_call(x):
            nonlocal calls_used, best_f, best_x
            if calls_used >= budget or check_deadline():
                return np.inf
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: CMA-ES for global search
        # Allocate ~65% of budget to CMA-ES
        cma_budget = int(budget * 0.65)
        
        # Initialize CMA-ES parameters
        x = np.zeros(n)
        sigma = 2.0  # Initial step size
        pop_size = max(10, int(4 + 3 * np.log(n)))  # Standard CMA-ES population size
        
        # CMA-ES state
        C = np.eye(n)  # Covariance matrix
        C_inv = np.eye(n)
        p_sigma = np.zeros(n)  # Evolution path for sigma
        p_c = np.zeros(n)  # Evolution path for covariance
        c_sigma = 1 / (n + 4)
        c_c = (2 + 2 / (n + 1)) / (n + 2 + 2 / (n + 1))
        mu = pop_size // 2  # Number of weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 2)) + c_sigma
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        # Evaluate initial point
        fx = do_call(x)
        if fx < best_f:
            best_f = fx
            best_x = x.copy()
        
        # CMA-ES loop
        cma_calls = 1
        while cma_calls < cma_budget and not check_deadline():
            # Generate population
            offspring = []
            for i in range(pop_size):
                z = rng.standard_normal(n)
                # Sample from N(x, sigma^2 * C)
                # Use Cholesky decomposition
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite, reset it
                    C = np.eye(n)
                    C_inv = np.eye(n)
                    p_sigma = np.zeros(n)
                    p_c = np.zeros(n)
                    L = np.eye(n)
                y = x + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                offspring.append(y)
            
            # Evaluate offspring
            offspring_f = []
            for y in offspring:
                fy = do_call(y)
                cma_calls += 1
                offspring_f.append(fy)
            
            # Sort by fitness
            order = np.argsort(offspring_f)
            sorted_offspring = [offspring[i] for i in order]
            sorted_offspring_f = [offspring_f[i] for i in order]
            
            # Update best
            if sorted_offspring_f[0] < best_f:
                best_f = sorted_offspring_f[0]
                best_x = sorted_offspring[0].copy()
            
            # Update mean
            x_new = np.sum(weights[:, np.newaxis] * np.array(sorted_offspring[:mu]), axis=0)
            
            # Update evolution paths
            y_weighted = (x_new - x) / (sigma + 1e-10)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_weighted
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_weighted
            
            # Update covariance matrix
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (cma_calls + 1)))) / chi_n < (1.4 + 2 / (n + 1))
            h_sigma = min(1.0, h_sigma) if h_sigma else 0.0
            
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                y_i = (sorted_offspring[i] - x) / (sigma + 1e-10)
                rank_mu += weights[i] * np.outer(y_i, y_i)
            
            C = (1 + c_c * (2 - c_c) - c_c) * C + c_c * np.outer(p_c, p_c) + c_mu * rank_mu
            # c_mu = c_c * (2 - c_c) - (1 - c_c) * (2 - c_c) ... actually:
            c_mu = 2 * (mu_eff - 2 + np.sum(weights ** 2)) / ((n + 2) ** 2 + mu_eff)
            
            # Ensure C is positive definite
            C = (C + C.T) / 2
            try:
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                C_inv = np.eye(n)
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)
            
            x = x_new
        
        # Phase 2: Local polish using Nelder-Mead from multiple starting points
        remaining_budget = budget - calls_used
        if remaining_budget > 20 and not check_deadline():
            polish_starts = [best_x.copy()]
            for scale in [0.3, 0.7, 1.5]:
                px = best_x + scale * rng.standard_normal(n)
                px = np.clip(px, -5, 5)
                polish_starts.append(px)
            
            for ps in polish_starts:
                if check_deadline() or calls_used >= budget - 5:
                    break
                
                remaining = budget - calls_used
                if remaining < 15:
                    break
                
                def objective(x):
                    nonlocal calls_used
                    if check_deadline() or calls_used >= budget:
                        raise ValueError("budget exceeded")
                    val = func(np.clip(x, -5, 5))
                    calls_used += 1
                    return val
                
                try:
                    res = minimize(
                        objective,
                        ps,
                        method='Nelder-Mead',
                        options={
                            'maxiter': min(remaining - 5, 300),
                            'xatol': 1e-8,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass
        
        # If no calls were made, make at least one
        if calls_used == 0:
            x = np.zeros(n)
            fx = do_call(x)
            best_x = x
            best_f = fx
        
        return best_x, best_f
