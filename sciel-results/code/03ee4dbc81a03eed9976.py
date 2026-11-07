"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer with multi-start strategy for BBOB f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # Safety margin before 300s limit
        calls = 0

        def can_continue():
            return calls < budget and time.time() < deadline

        def remaining():
            return budget - calls

        def make_func_counter():
            nonlocal calls
            def f(x):
                nonlocal calls
                calls += 1
                return func(x)
            return f

        f_count = make_func_counter()

        best_x = None
        best_f = float('inf')

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_x = x.copy()
                best_f = fx

        rng = np.random.default_rng(42)

        # CMA-ES Implementation
        def cma_es(x_start, sigma_start, max_calls):
            nonlocal calls, best_x, best_f, f_count
            
            n = dim
            # Population size
            lambda_ = max(4, int(4 + 3 * np.log(n)))
            # CMA-ES parameters
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
            
            # State
            x_mean = x_start.copy()
            sigma = sigma_start
            C = np.eye(n)
            C_inv = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            # Precompute Cholesky of C
            try:
                C_eig = np.linalg.cholesky(C)
            except:
                C_eig = np.eye(n)
                
            # Standard normal distribution
            z = rng.standard_normal((lambda_, n))
            
            # Initial evaluation
            # We don't evaluate x_start here if it was already evaluated, 
            # but for simplicity in multi-start, we assume fresh start or 
            # we pass in the best known. 
            # Actually, to save calls, we should check if x_start is best_x.
            # However, for CMA-ES to work well, it needs a population.
            # We will evaluate the first population.
            
            fitnesses = []
            xs = []
            
            for step in range(max_calls // lambda_ + 1):
                if not can_continue():
                    break
                
                # Generate population
                try:
                    B = np.linalg.cholesky(C)
                except:
                    # Regularization if C becomes singular
                    C = np.eye(n)
                    B = np.eye(n)
                    
                # Sample
                for i in range(lambda_):
                    if not can_continue():
                        break
                    y = B @ rng.standard_normal(n)
                    x_new = x_mean + sigma * y
                    x_new = np.clip(x_new, -5, 5)
                    f_new = f_count(x_new)
                    xs.append(x_new)
                    fitnesses.append(f_new)
                    
                    if f_new < best_f:
                        best_x = x_new.copy()
                        best_f = f_new
                        
                if len(xs) < lambda_:
                    break
                    
                # Sort by fitness
                idx = np.argsort(fitnesses)
                xs = np.array(xs)
                fitnesses = np.array(fitnesses)
                xs = xs[idx]
                
                # Update mean
                x_mean = np.sum(weights[:, np.newaxis] * xs[:mu], axis=0)
                
                # Update sigma
                # Calculate step length
                y = x_mean - x_mean  # This is 0, wait.
                # Standard CMA-ES update:
                # x_old_mean is needed. Let's store it.
                
                # I need to restructure the loop to keep track of old mean.
                # Let's restart the CMA-ES logic properly.
                pass

        # Corrected CMA-ES Implementation
        def run_cma_es(x_init, sigma_init, max_calls):
            nonlocal calls, best_x, best_f, f_count
            
            n = dim
            lambda_ = max(4, int(4 + 3 * np.log(n)))
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
            
            x_mean = x_init.copy()
            sigma = sigma_init
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            # Initial population evaluation
            # We will generate population in each iteration
            
            while can_continue() and max_calls > 0:
                # Generate population
                xs = []
                fs = []
                
                # Calculate B from C
                try:
                    B = np.linalg.cholesky(C)
                except:
                    B = np.eye(n)
                    
                for i in range(lambda_):
                    if not can_continue() or max_calls <= 0:
                        break
                    z = rng.standard_normal(n)
                    y = B @ z
                    x_new = x_mean + sigma * y
                    x_new = np.clip(x_new, -5, 5)
                    f_new = f_count(x_new)
                    max_calls -= 1
                    xs.append(x_new)
                    fs.append(f_new)
                    
                    if f_new < best_f:
                        best_x = x_new.copy()
                        best_f = f_new
                        
                if len(xs) < lambda_:
                    break
                    
                xs = np.array(xs)
                fs = np.array(fs)
                
                # Sort
                idx = np.argsort(fs)
                xs = xs[idx]
                fs = fs[idx]
                
                # Update mean
                x_old_mean = x_mean.copy()
                x_mean = np.sum(weights[:, np.newaxis] * xs[:mu], axis=0)
                
                # Update ps
                # Calculate step in C space
                try:
                    C_inv_sqrt = np.linalg.inv(B)
                except:
                    C_inv_sqrt = np.eye(n)
                    
                y_vec = C_inv_sqrt @ (x_mean - x_old_mean) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_vec
                
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2) - np.sqrt(n) * (1 - 1 / np.sqrt(2))))
                
                # Update pc
                h_sig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2) < (1 + 2 / (n + 1)) * np.sqrt(n))
                pc = (1 - c_c) * pc + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_vec
                
                # Update C
                delta_h = (1 - h_sig) * c_c * (2 - c_c)
                rank_one = np.outer(pc, pc)
                rank_mu = np.sum(weights[:, np.newaxis] * np.outer(xs[:mu] - x_old_mean, xs[:mu] - x_old_mean) / (sigma**2), axis=0)
                
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
                
                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2
                # Simple regularization if needed
                try:
                    np.linalg.cholesky(C)
                except:
                    C = 0.1 * C + 0.9 * np.eye(n)
                    C = (C + C.T) / 2

        # Strategy:
        # 1. Generate a set of diverse starting points.
        # 2. Run CMA-ES from the best few, splitting the budget.
        
        starts = []
        # Random uniform
        for _ in range(5):
            starts.append(rng.uniform(-5, 5, dim))
        # Near origin
        starts.append(np.zeros(dim))
        # Random normal
        starts.append(rng.normal(0, 2, dim))
        # Corners
        starts.append(np.array([5.0] * dim))
        starts.append(np.array([-5.0] * dim))
        
        # Evaluate starts
        start_results = []
        for x0 in starts:
            if not can_continue():
                break
            x0 = np.clip(x0, -5, 5)
            fx = f_count(x0)
            update_best(x0, fx)
            start_results.append((x0, fx))
            
        if not can_continue():
            return best_x, best_f
            
        # Sort starts
        start_results.sort(key=lambda t: t[1])
        
        # Allocate budget to top 3 starts
        # 1st: 50%, 2nd: 30%, 3rd: 20%
        allocations = [0.5, 0.3, 0.2]
        
        for i in range(min(3, len(start_results))):
            if not can_continue():
                break
            x0, fx0 = start_results[i]
            alloc = int(remaining() * allocations[i])
            # Ensure at least one population fits
            if alloc < 4:
                continue
            run_cma_es(x0, 1.0, alloc)
            
        # Final local refinement if budget remains
        if can_continue() and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            n_refine = remaining()
            for i in range(n_refine):
                if not can_continue():
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = f_count(y)
                if fy < fx:
                    x = y
                    fx = fy
                    update_best(x, fx)
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                    if sigma < 0.01:
                        break
                        
        return best_x, best_f

# EVOLVE-BLOCK-END
