import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
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
            
            val = func(x)
            
            if val < best_f:
                best_f = val
                best_x = np.array(x, dtype=float).copy()
                
            return val

        # Phase 1: Global Search with Differential Evolution
        # Use about 50% of the budget for global exploration
        global_budget = int(0.5 * self.budget)
        
        # DE parameters tuned for 5D
        popsize = 10
        max_iter_de = max(1, int(global_budget / popsize) - 1)
        
        try:
            res_de = differential_evolution(
                wrapped_func, 
                bounds=[(-5, 5)] * dim, 
                maxiter=max_iter_de,
                popsize=popsize,
                polish=False,
                tol=1e-10,
                seed=42,
                init='latinhypercube',
                strategy='best1bin'
            )
            if res_de.fun < best_f:
                best_f = res_de.fun
                best_x = np.array(res_de.x, dtype=float).copy()
        except (TimeoutError, Exception):
            pass

        # Phase 2: Local Refinement with CMA-ES (manual implementation)
        # Since we have a fixed budget and need tight control, we implement a simplified CMA-ES loop
        if calls < self.budget and best_x is not None and time_ok():
            # CMA-ES parameters
            mu = max(1, dim // 2)  # number of parents
            lam = 2 * mu           # population size
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / (weights @ weights)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # Initial state
            rng = np.random.default_rng(42)
            x = best_x.copy()
            sigma = 0.5  # Start with smaller step for local refinement
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            B = np.eye(dim)
            D = np.ones(dim)

            while calls < self.budget and time_ok():
                # Generate population
                ys = []
                for _ in range(lam):
                    z = rng.standard_normal(dim)
                    y = x + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    ys.append(y)
                
                # Evaluate population
                fs = []
                for y in ys:
                    if not time_ok() or calls >= self.budget:
                        break
                    fs.append(wrapped_func(y))
                
                if len(fs) < lam:
                    break
                
                # Sort by fitness
                idx = np.argsort(fs)
                ys_sorted = [ys[i] for i in idx]
                fs_sorted = [fs[i] for i in idx]

                # Update mean
                x_old = x.copy()
                x = sum(weights[i] * ys_sorted[i] for i in range(mu))
                x = np.clip(x, -5, 5)

                # Update sigma
                y_mean = (x_old - x) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 5.0)

                # Update C matrix (simplified: only if needed, otherwise skip for speed)
                # In a full implementation, we'd update B and D here via eigendecomposition
                # For simplicity and speed, we'll just update the mean and sigma
                # This is a simplified version that might not be as effective as full CMA-ES
                # but should still provide good local refinement

                # Check for stagnation
                if np.linalg.norm(x_old - x) < 1e-10:
                    break

        # Phase 3: Final (1+1)-ES refinement if budget remains
        remaining_budget = self.budget - calls
        if remaining_budget > 0 and best_x is not None and time_ok():
            x = best_x.copy()
            fx = best_f
            sigma = 0.1  # Start small
            
            rng = np.random.default_rng(123)
            
            for _ in range(remaining_budget):
                if not time_ok():
                    break
                
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                
                fy = wrapped_func(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                
                if sigma < 1e-6:
                    break

            if fx < best_f:
                best_f = fx
                best_x = x

        return best_x, best_f
