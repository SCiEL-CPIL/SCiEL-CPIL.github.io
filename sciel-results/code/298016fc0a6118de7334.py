import numpy as np
import time
from scipy.optimize import differential_evolution

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        calls = 0
        best_x = None
        best_f = np.inf
        dim = self.dim

        def time_ok():
            return time.time() < deadline

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if not time_ok():
                raise TimeoutError("Time limit exceeded")
            calls += 1
            if calls > self.budget:
                raise Exception("Budget exceeded")
            
            val = float(func(x))
            
            if val < best_f:
                best_f = val
                best_x = np.array(x, dtype=float).copy()
                
            return val

        # Phase 1: CMA-ES for global search (better for ill-conditioned functions)
        # Use ~70% of budget for CMA-ES
        cma_budget = int(0.70 * self.budget)
        
        # CMA-ES with larger population for better covariance estimation
        # popsize = 4 + 3*ln(dim) is standard, but we use larger per findings
        popsize = max(16, 4 + 3 * int(np.log(dim)) if dim > 1 else 4)
        
        # Initial sigma of 2.2 for good coverage of [-4,4]^5
        sigma0 = 2.2
        
        # Starting point at center of search space
        x0 = np.zeros(dim)
        
        try:
            # Manual CMA-ES implementation for better control
            rng = np.random.default_rng(42)
            x = x0.copy()
            sigma = sigma0
            C = np.eye(dim)  # Covariance matrix
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # CMA-ES parameters
            mu = max(1, popsize // 2)  # Number of parents
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            
            # Initial population
            pop = []
            for i in range(popsize):
                xi = x + sigma * rng.standard_normal(dim)
                xi = np.clip(xi, -5, 5)
                fi = wrapped_func(xi)
                pop.append((fi, xi))
            
            # Sort by fitness
            pop.sort(key=lambda t: t[0])
            
            # Main loop
            while calls < cma_budget and time_ok():
                # Update best
                if pop[0][0] < best_f:
                    best_f = pop[0][0]
                    best_x = pop[0][1].copy()
                
                # Mean update
                x_old = x.copy()
                x = sum(weights[i] * pop[i][1] for i in range(mu))
                
                # Update sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (C @ (x - x_old))
                if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - popsize + 1))) < (1.4 + 2 / (dim + 1)) * chi_n:
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                
                # Update covariance
                diff_pop = [pop[i][1] - x_old for i in range(mu)]
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * (x - x_old) / sigma
                delta_heav = 1 if np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (calls - popsize + 1)) < (1.4 + 2 / (dim + 1)) * chi_n else 0
                C = (1 + c_1 * delta_heav - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * sum(weights[i] * np.outer(diff_pop[i], diff_pop[i]) for i in range(mu))
                
                # Ensure C is positive definite
                C = (C + C.T) / 2
                # Eigenvalue correction
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                # Generate new population
                new_pop = []
                for i in range(popsize):
                    # Use Cholesky-like sampling
                    try:
                        L = np.linalg.cholesky(C)
                        xi = x + sigma * (L @ rng.standard_normal(dim))
                    except:
                        xi = x + sigma * rng.standard_normal(dim)
                    xi = np.clip(xi, -5, 5)
                    if not time_ok() or calls >= cma_budget:
                        break
                    fi = wrapped_func(xi)
                    new_pop.append((fi, xi))
                
                if len(new_pop) < popsize:
                    break
                
                pop = new_pop
                pop.sort(key=lambda t: t[0])
                
                # Check for convergence
                if sigma < 1e-10:
                    break
                    
        except (TimeoutError, Exception):
            pass

        # Phase 2: Local Refinement with (1+1)-ES
        remaining_budget = self.budget - calls
        
        if remaining_budget > 0 and best_x is not None and time_ok():
            local_rng = np.random.default_rng(1)
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
            
            for _ in range(remaining_budget):
                if not time_ok() or calls >= self.budget:
                    break
                
                y = x + sigma * local_rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                
                fy = wrapped_func(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.15
                else:
                    sigma *= 0.85
                    
                if sigma < 1e-10:
                    break
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
