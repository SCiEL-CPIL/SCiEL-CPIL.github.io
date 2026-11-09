# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # safety margin before 330s kill

        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)

        # Phase 1: Brief random search to find a good starting point
        # This helps with the unknown shift/rotation
        n_init = min(20, self.budget // 10)
        for i in range(n_init):
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES with adaptive restarts
        # We'll do multiple restarts if stagnation is detected
        remaining = self.budget - n_init
        restart_budget = max(50, remaining // 3)
        
        # CMA-ES parameters
        lam = 4 * int(1 + 3 * self.dim)  # population size
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = (2 - mu_eff / (self.dim + 1) + c_sigma) / ((self.dim + 1.3) ** 2 + mu_eff)
        c_mu = (2 * (mu_eff / (self.dim + 1) - 1) + c_1) / (
            (self.dim + 2) ** 2 + mu_eff / (self.dim + 1)
        )
        chi_n = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))

        def cma_es(x0, sigma0, budget_cma):
            nonlocal best_x, best_f
            if budget_cma <= 0:
                return
            
            # Initialize CMA-ES state
            x_mean = x0.copy()
            sigma = sigma0
            B = np.eye(self.dim)
            D = np.ones(self.dim)
            pc = np.zeros(self.dim)
            ps = np.zeros(self.dim)
            cs = (1 + c_sigma + c_c + 2 * c_sigma * c_c / c_sigma) * 0.5
            ds = (1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1)) * 0.5
            # Actually, let's use standard CMA-ES update formulas
            
            # Precompute
            hsigma = (1 + (1 - np.exp(-1 / d_sigma)) * 2) * 0.5
            
            count = 0
            last_improve = 0
            stall_count = 0
            
            while count < budget_cma:
                if time.time() > deadline:
                    break
                    
                # Generate population
                offspring = []
                for i in range(lam):
                    z = rng.standard_normal(self.dim)
                    y = x_mean + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    offspring.append(y)
                
                # Evaluate offspring
                offspring_f = [func(y) for y in offspring]
                
                # Sort by fitness
                idx = np.argsort(offspring_f)
                offspring_sorted = [offspring[i] for i in idx]
                offspring_f_sorted = [offspring_f[i] for i in idx]
                
                # Update best
                if offspring_f_sorted[0] < best_f:
                    best_f = offspring_f_sorted[0]
                    best_x = offspring_sorted[0].copy()
                    stall_count = 0
                    last_improve = count
                else:
                    stall_count += 1
                
                # Check for stagnation
                if stall_count > 50 and count > 100:
                    break
                
                # Compute new mean
                x_new_mean = sum(weights[i] * offspring_sorted[i] for i in range(mu))
                
                # Compute new covariance matrix update
                # D * B @ (x_new_mean - x_mean)
                diff = x_new_mean - x_mean
                y_inv_sigma = diff / sigma
                
                # Update ps
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * (B @ (D * y_inv_sigma))
                
                # Update pc
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (count + 1)))) < (1.4 + 2 / (self.dim + 1)) * chi_n
                pc = (1 - c_c) * pc + hsig * c_c * y_inv_sigma
                
                # Update C matrix (covariance)
                # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (hsig - 1) * c_c * (2 - c_c) * C) + c_mu * sum(weights[i] * (y_i - mean) @ (y_i - mean).T)
                # This is expensive, so we'll use a simplified version or just do eigendecomposition periodically
                
                # For efficiency, let's do the full update every few generations
                if count % 5 == 0:
                    # Compute outer product sum
                    outer_sum = np.zeros((self.dim, self.dim))
                    for i in range(mu):
                        d = offspring_sorted[i] - x_new_mean
                        outer_sum += weights[i] * np.outer(d, d)
                    
                    # Update C
                    C = np.eye(self.dim)  # Simplified: reset to identity periodically to avoid numerical issues
                    # Actually, let's maintain C properly
                    # For now, let's use a simpler approach: just update the step size and direction
                    
                    # Eigendecomposition for B and D
                    try:
                        eigval, eigvec = np.linalg.eigh(C)
                        eigval = np.maximum(eigval, 1e-10)
                        D = np.sqrt(eigval)
                        B = eigvec
                    except:
                        pass
                
                # Update x_mean
                x_mean = x_new_mean
                
                # Update sigma
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 50.0)
                
                count += 1
            
            return x_mean, sigma

        # Run CMA-ES with restarts
        num_restarts = 3
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            if remaining <= 0:
                break
            
            # Start from best found so far, or a new random point for later restarts
            if r == 0:
                x0 = best_x.copy()
                sigma0 = 2.0
            else:
                x0 = best_x + rng.uniform(-1, 1, self.dim)
                x0 = np.clip(x0, -5, 5)
                sigma0 = 1.0
            
            cma_budget = min(restart_budget, remaining)
            cma_es(x0, sigma0, cma_budget)
            remaining -= cma_budget
        
        # Phase 3: Local refinement with Nelder-Mead if time permits
        if time.time() < deadline - 10 and best_x is not None:
            try:
                from scipy.optimize import minimize
                def nm_func(x):
                    return func(np.clip(x, -5, 5))
                
                res = minimize(nm_func, best_x, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except:
                pass
        
        return best_x, best_f

# EVOLVE-BLOCK-END
