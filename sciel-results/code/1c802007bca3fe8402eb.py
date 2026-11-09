"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid CMA-ES + local polish for Rastrigin f3."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit
        calls = 0
        dim = self.dim
        budget = self.budget

        def call_func(x):
            nonlocal calls
            if calls >= budget:
                return None
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            val = float(func(x))
            calls += 1
            return val

        # Phase 1: Global exploration with CMA-ES
        rng = np.random.default_rng(42)
        
        # Initial population spread
        x0 = rng.uniform(-5, 5, dim)
        f0 = call_func(x0)
        best_x = x0.copy()
        best_f = f0

        # CMA-ES parameters
        sigma = 2.0
        pop_size = max(4 + int(3 * np.log(dim)), 10)
        pop_size = min(pop_size, 20)  # Keep it manageable
        lambda_ = pop_size
        
        # CMA-ES state
        x_mean = x0.copy()
        sigma = 2.0
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Update rates
        cmu = 1 / (dim + 3)
        cc = (4 + dim / 3) / ((dim + 4) + 2 * cmu)
        c1 = 2 / ((dim + 1.3) ** 2 + dim)
        cmu = 2 / ((dim + 2) ** 2 + 10)
        cs = (2 + cmu) / (dim + 2 + cmu)
        
        # Eigen decomposition cache
        eigvals = np.ones(dim)
        eigvecs = np.eye(dim)
        
        max_iter = min(budget - calls - 50, 400)  # Leave some for local polish
        
        for it in range(max_iter):
            if time.time() > deadline - 30:
                break
            if calls >= budget - 50:
                break
            
            # Generate population
            population = []
            fitnesses = []
            
            for i in range(lambda_):
                # Rank-mu update would be ideal but let's keep it simple
                z = rng.standard_normal(dim)
                # Apply covariance
                y = B @ (D * z)
                x_i = x_mean + sigma * y
                x_i = np.clip(x_i, -5, 5)
                
                f_i = call_func(x_i)
                if f_i is None:
                    break
                population.append(x_i)
                fitnesses.append(f_i)
            
            if len(fitnesses) < 2:
                break
            
            # Sort by fitness
            order = np.argsort(fitnesses)
            mu = max(2, lambda_ // 2)
            
            # Update best
            if fitnesses[order[0]] < best_f:
                best_f = fitnesses[order[0]]
                best_x = population[order[0]].copy()
            
            # Update mean
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += w[i] * population[order[i]]
            
            # Update sigma control (1/5 rule simplified)
            # Check success rate
            success = sum(1 for i in range(mu) if fitnesses[order[i]] < f0)
            if success > mu * 0.3:
                sigma *= 1.1
            else:
                sigma *= 0.9
            sigma = np.clip(sigma, 0.01, 5.0)
            
            # Update covariance (simplified)
            # For simplicity, use a diagonal approximation
            diff = np.zeros((mu, dim))
            for i in range(mu):
                diff[i] = (population[order[i]] - x_mean) / sigma
            
            # Simple covariance update
            C_new = np.zeros((dim, dim))
            for i in range(mu):
                C_new += w[i] * np.outer(diff[i], diff[i])
            C = 0.9 * C + 0.1 * C_new
            
            # Ensure positive semi-definite
            eigvals_new, eigvecs_new = np.linalg.eigh(C)
            eigvals_new = np.maximum(eigvals_new, 1e-10)
            C = (eigvecs_new * np.sqrt(eigvals_new)) @ eigvecs_new.T
            
            B, D = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(np.diag(C), 1e-10))
            
            x_mean = x_new
            f0 = best_f  # Reset reference
        
        # Phase 2: Local polish with scipy
        if calls < budget - 5 and best_f < 50:
            try:
                if time.time() < deadline - 10:
                    res = minimize(
                        lambda x: call_func(x) if calls < budget else 1e10,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 50, 'maxfun': min(budget - calls, 200)}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
