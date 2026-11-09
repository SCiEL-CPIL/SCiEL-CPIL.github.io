"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer for BBOB f4 (Büche-Rastrigin).
    
    Phase 1: Coarse random sampling to find promising basins.
    Phase 2: CMA-ES with restarts on the most promising basin.
    Phase 3: Focused CMA-ES on the best solution found.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin out of 330s limit
        
        rng = np.random.default_rng(42)
        
        # Track best solution found
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        def check_time():
            return time.time() >= deadline
        
        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return best_f
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx
        
        # Phase 1: Coarse random sampling (50 evaluations)
        n_coarse = min(50, self.budget)
        coarse_points = []
        for i in range(n_coarse):
            if check_time():
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = safe_call(x)
            coarse_points.append((fx, x))
        
        if best_x is None:
            return np.zeros(self.dim), 0.0
        
        # Sort coarse points by fitness
        coarse_points.sort(key=lambda t: t[0])
        
        # Phase 2: CMA-ES with restarts on top basins
        # Use remaining budget for CMA-ES
        remaining = self.budget - calls_made
        
        # Select top basins for CMA-ES
        n_basins = min(3, len(coarse_points))
        basin_starts = [coarse_points[i][1] for i in range(n_basins)]
        
        # Budget allocation: give more to the best basin
        if n_basins == 1:
            budgets = [remaining]
        elif n_basins == 2:
            budgets = [int(remaining * 0.6), int(remaining * 0.4)]
        else:
            budgets = [int(remaining * 0.5), int(remaining * 0.3), int(remaining * 0.2)]
        
        for i, (start_x, budget_i) in enumerate(zip(basin_starts, budgets)):
            if check_time() or calls_made >= self.budget or budget_i <= 0:
                break
            
            # CMA-ES implementation
            x = start_x.copy()
            sigma = 1.0  # initial step size
            popsize = max(4, self.dim + 2)  # population size
            
            # CMA-ES state
            C = np.eye(self.dim)  # covariance matrix
            C_inv = np.eye(self.dim)  # inverse covariance
            C_sqrt = np.eye(self.dim)  # square root of C
            C_inv_sqrt = np.eye(self.dim)  # inverse square root
            
            # Evolution paths
            p_sigma = np.zeros(self.dim)
            p_c = np.zeros(self.dim)
            
            # Parameters
            c_sigma = (popsize + 2) / (self.dim + 4)
            mu = popsize // 2  # number of parents
            weights = np.ones(mu)
            weights -= np.arange(mu) * 0.1
            weights = np.maximum(weights, 0.1)
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
            
            # Initial step size based on basin quality
            if i == 0:
                sigma = 1.5
            else:
                sigma = 1.0
            
            for gen in range(budget_i):
                if check_time() or calls_made + popsize > self.budget:
                    break
                
                # Generate population
                offspring = []
                for j in range(popsize):
                    # Sample from multivariate normal
                    z = rng.standard_normal(self.dim)
                    y = x + sigma * C_sqrt @ z
                    y = np.clip(y, -5, 5)
                    offspring.append(y)
                
                # Evaluate offspring
                offspring_f = []
                for y in offspring:
                    fy = safe_call(y)
                    offspring_f.append(fy)
                
                if check_time():
                    break
                
                # Sort by fitness
                idx = np.argsort(offspring_f)
                offspring = [offspring[i] for i in idx]
                offspring_f = [offspring_f[i] for i in idx]
                
                # Update mean
                x_new = np.sum(weights[:, None] * np.array(offspring[:mu]), axis=0)
                
                # Update evolution paths
                # Compute C_inv_sqrt @ (x_new - x)
                diff = x_new - x
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ diff / sigma
                
                # Sigma update
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)
                
                # C evolution path
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt @ diff / sigma
                
                # Update covariance
                delta_heav = (1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1)) * c_c * (2 - c_c)
                C = (1 + c_c * (2 - c_c) - c_c) * C + c_c * (np.outer(p_c, p_c) - delta_heav * C)
                
                # For efficiency, update C_sqrt periodically
                if gen % 10 == 0:
                    # Recompute C_inv_sqrt
                    try:
                        C_inv = np.linalg.inv(C)
                        C_inv_sqrt = np.linalg.cholesky(C_inv).T
                        C_sqrt = np.linalg.cholesky(C)
                    except:
                        # Reset if numerical issues
                        C = np.eye(self.dim)
                        C_inv = np.eye(self.dim)
                        C_sqrt = np.eye(self.dim)
                        C_inv_sqrt = np.eye(self.dim)
                
                x = x_new
            
            # Update global best
            if best_x is None or best_f > np.min(offspring_f):
                pass  # best_f already updated in safe_call
        
        # Phase 3: Final focused CMA-ES on the best solution
        if not check_time() and calls_made < self.budget:
            remaining = self.budget - calls_made
            if remaining > 10:
                x = best_x.copy()
                sigma = 0.5
                popsize = max(4, self.dim + 2)
                
                C = np.eye(self.dim)
                C_inv = np.eye(self.dim)
                C_sqrt = np.eye(self.dim)
                C_inv_sqrt = np.eye(self.dim)
                p_sigma = np.zeros(self.dim)
                p_c = np.zeros(self.dim)
                
                c_sigma = (popsize + 2) / (self.dim + 4)
                mu = popsize // 2
                weights = np.ones(mu)
                weights -= np.arange(mu) * 0.1
                weights = np.maximum(weights, 0.1)
                weights /= weights.sum()
                mu_eff = 1.0 / np.sum(weights ** 2)
                c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
                d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
                
                for gen in range(remaining):
                    if check_time() or calls_made + popsize > self.budget:
                        break
                    
                    offspring = []
                    for j in range(popsize):
                        z = rng.standard_normal(self.dim)
                        y = x + sigma * C_sqrt @ z
                        y = np.clip(y, -5, 5)
                        offspring.append(y)
                    
                    offspring_f = []
                    for y in offspring:
                        fy = safe_call(y)
                        offspring_f.append(fy)
                    
                    if check_time():
                        break
                    
                    idx = np.argsort(offspring_f)
                    offspring = [offspring[i] for i in idx]
                    offspring_f = [offspring_f[i] for i in idx]
                    
                    x_new = np.sum(weights[:, None] * np.array(offspring[:mu]), axis=0)
                    
                    diff = x_new - x
                    p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ diff / sigma
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                    sigma = np.clip(sigma, 1e-8, 5.0)
                    
                    p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt @ diff / sigma
                    delta_heav = (1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1)) * c_c * (2 - c_c)
                    C = (1 + c_c * (2 - c_c) - c_c) * C + c_c * (np.outer(p_c, p_c) - delta_heav * C)
                    
                    if gen % 10 == 0:
                        try:
                            C_inv = np.linalg.inv(C)
                            C_inv_sqrt = np.linalg.cholesky(C_inv).T
                            C_sqrt = np.linalg.cholesky(C)
                        except:
                            C = np.eye(self.dim)
                            C_inv = np.eye(self.dim)
                            C_sqrt = np.eye(self.dim)
                            C_inv_sqrt = np.eye(self.dim)
                    
                    x = x_new
        
        return best_x, best_f
# EVOLVE-BLOCK-END
