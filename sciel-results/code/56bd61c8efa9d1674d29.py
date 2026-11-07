"""CMA-ES based optimizer for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
    with a final local polish phase.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        dim = self.dim
        budget = self.budget

        calls_made = 0
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            try:
                x = np.asarray(x, dtype=np.float64)
                if x.shape != (dim,):
                    x = x.reshape(dim)
                x = np.clip(x, -5.0, 5.0)
                f = float(func(x))
                if not np.isfinite(f):
                    f = 1e10
                calls_made += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                return f
            except Exception:
                return None

        # --- Phase 1: CMA-ES ---
        # CMA-ES parameters
        # Population size lambda
        lambda_pop = 4 + int(3 * np.log(dim))  # Standard rule of thumb
        lambda_pop = max(lambda_pop, 10) # Ensure at least 10 for stability
        
        # Initial mean and sigma
        x_mean = np.zeros(dim)
        sigma = 2.0  # Initial step size to cover the [-5, 5] box reasonably
        
        # CMA-ES state variables
        # We will implement a simplified CMA-ES update logic manually 
        # to avoid dependency on external CMA-ES libraries if not strictly necessary, 
        # but scipy does not have a built-in CMA-ES. 
        # However, implementing full CMA-ES is complex. 
        # Alternative: Use a robust derivative-free method like Powell or Nelder-Mead 
        # with multiple restarts? 
        # Or, use a simplified version of CMA-ES.
        
        # Given the constraint "Python standard library, NumPy and SciPy are available", 
        # and no specific CMA-ES library mentioned, I will implement a simplified 
        # CMA-ES-like strategy or use a well-tuned Nelder-Mead with restarts.
        
        # Actually, let's implement a basic CMA-ES.
        # Key components:
        # 1. Sample lambda points from N(x_mean, sigma^2 * C)
        # 2. Evaluate
        # 3. Sort by fitness
        # 4. Update x_mean using weighted average of top mu points
        # 5. Update C and sigma
        
        # For dim=5, a simplified CMA-ES is feasible.
        
        # Weights
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / dim) / ((dim + 2) ** 2 + mu_eff))
        
        # State
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        inv_C = np.eye(dim)
        # Precompute square root of C for sampling
        # We need to update sqrt(C) each iteration
        
        # Helper to get sqrt(C)
        def get_sqrt_C(C):
            # Eigen decomposition
            vals, vecs = np.linalg.eigh(C)
            vals = np.maximum(vals, 1e-10) # Ensure positive
            sqrt_vals = np.sqrt(vals)
            return vecs @ np.diag(sqrt_vals) @ vecs.T

        sqrt_C = get_sqrt_C(C)
        
        # Budget allocation
        # Use ~80% of budget for CMA-ES, 20% for polish
        cma_budget = int(budget * 0.8)
        
        # Initial evaluation of mean
        f_mean = safe_call(x_mean)
        if f_mean is not None:
            X = [x_mean]
            Y = [f_mean]
        else:
            X = []
            Y = []

        for gen in range(1000): # Max generations, will break on budget/time
            if calls_made >= cma_budget:
                break
            if time.time() > deadline - 5.0: # Leave time for polish
                break
                
            # 1. Sample population
            pop = []
            for i in range(lambda_pop):
                z = np.random.randn(dim)
                x_new = x_mean + sigma * (sqrt_C @ z)
                pop.append(x_new)
                
            # 2. Evaluate
            fitnesses = []
            for x in pop:
                f = safe_call(x)
                if f is None:
                    f = 1e10
                fitnesses.append(f)
                X.append(x)
                Y.append(f)
            
            # 3. Sort
            indices = np.argsort(fitnesses)
            sorted_pop = [pop[i] for i in indices]
            sorted_fit = [fitnesses[i] for i in indices]
            
            # Update best
            if sorted_fit[0] < best_f:
                best_f = sorted_fit[0]
                best_x = sorted_pop[0].copy()

            # 4. Update mean
            x_new_mean = np.zeros(dim)
            for i in range(mu):
                x_new_mean += weights[i] * sorted_pop[i]
            
            # 5. Update sigma control path
            # Normalize step size
            # d_sigma is 1.0 for simplicity
            d_sigma = 1.0
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (sqrt_C.T @ (x_new_mean - x_mean)) / sigma
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt(dim):
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(dim) - 1))
            
            # 6. Update covariance matrix control path
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new_mean - x_mean) / sigma
            
            # Rank-one and rank-mu update
            delta_sigma = (1 - (1 - c_sigma) ** (gen + 1)) * 1.0 # Approximation
            hsig = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt(dim) else 0.0
            
            # C update
            C = (1 + c_1 * hsig - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c)
            for i in range(mu):
                y_i = (sorted_pop[i] - x_mean) / sigma
                C += c_mu * weights[i] * np.outer(y_i, y_i)
                
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update inverse and sqrt
            try:
                inv_C = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                inv_C = np.eye(dim)
            sqrt_C = get_sqrt_C(C)
            
            x_mean = x_new_mean

        # --- Phase 2: Local Polish ---
        if best_x is not None and calls_made < budget and time.time() < deadline:
            # Use remaining budget for a high-precision local search
            # Nelder-Mead is good for final polish
            remaining_budget = budget - calls_made
            if remaining_budget > 5:
                try:
                    # Run Nelder-Mead from the best point
                    # We need to be careful with the call count
                    # scipy minimize does not have a strict callback count limit, 
                    # so we wrap the function to stop if we exceed budget.
                    
                    polish_calls = 0
                    def polish_func(x):
                        nonlocal polish_calls
                        if polish_calls >= remaining_budget:
                            return 1e10
                        f = safe_call(x)
                        if f is None:
                            return 1e10
                        polish_calls += 1
                        return f

                    res = minimize(polish_func, best_x, method='Nelder-Mead',
                                   options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
                    if res.fun < best_f and np.isfinite(res.fun):
                        best_x = np.clip(res.x, -5, 5)
                        best_f = res.fun
                except Exception:
                    pass

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_call(best_x)
            if best_f is None:
                best_f = 1e10

        return best_x, best_f
# EVOLVE-BLOCK-END
