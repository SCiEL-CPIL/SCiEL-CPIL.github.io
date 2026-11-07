"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with population-based search and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit

        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        best_x = None
        best_f = float('inf')
        calls = 0

        def check_time():
            return time.time() < deadline

        # Multi-start with CMA-ES
        # We'll use a simple CMA-ES implementation with restarts
        
        # Phase 1: Quick exploration with multiple random starts
        # Phase 2: Focused CMA-ES around the best point found
        
        # First, do a quick random search to find a good starting point
        n_random = min(50, budget // 10)
        for i in range(n_random):
            if calls >= budget or not check_time():
                break
            x = rng.uniform(-5, 5, d)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Now run CMA-ES from the best point
        if best_x is None:
            best_x = np.zeros(d)
            best_f = func(best_x)
            calls += 1

        # CMA-ES parameters
        # Population size
        lambda_pop = max(4, int(4 + 3 * np.log(d)))
        
        # CMA-ES state
        x_mean = best_x.copy()
        sigma = 2.0
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        # Adaptation parameters
        cc = (4 + lambda_pop / d) / (d + 4 + 2 * lambda_pop / d)
        cs = (lambda_pop + 2) / (d + lambda_pop + 2)
        cmu = min(2 - cc + 2 / (d * lambda_pop - 2), 
                  0.5 / (1 - cc))
        c_sigma = (lambda_pop + 2) / (d + 2)
        d_sigma = 1 + 2 * max(0, np.sqrt((lambda_pop - 1) / (d + 1)) - 1) + c_sigma
        
        # Step size control
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d * d))
        
        # Stagnation detection
        stagnation_count = 0
        stagnation_limit = 30
        best_f_in_phase = best_f
        
        max_generations = 200
        for gen in range(max_generations):
            if calls >= budget or not check_time():
                break
            
            # Generate population
            offspring = []
            offspring_f = []
            
            for i in range(lambda_pop):
                if calls >= budget or not check_time():
                    break
                
                # Sample from multivariate normal
                v = rng.standard_normal(d)
                # Apply covariance matrix
                # Cholesky decomposition for C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(d)
                
                y = x_mean + sigma * (L @ v)
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls += 1
                
                offspring.append(y)
                offspring_f.append(fy)
            
            if len(offspring) == 0:
                break
            
            # Sort offspring by fitness
            idx = np.argsort(offspring_f)
            offspring = [offspring[i] for i in idx]
            offspring_f = [offspring_f[i] for i in idx]
            
            # Update best
            if offspring_f[0] < best_f:
                best_f = offspring_f[0]
                best_x = offspring[0].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            if offspring_f[0] < best_f_in_phase:
                best_f_in_phase = offspring_f[0]
            
            # Check for restart
            if stagnation_count > stagnation_limit:
                # Restart with new random start
                x_mean = rng.uniform(-4, 4, d)
                sigma = 2.0
                C = np.eye(d)
                pc = np.zeros(d)
                ps = np.zeros(d)
                stagnation_count = 0
                best_f_in_phase = best_f
                continue
            
            # Update mean
            mu = lambda_pop // 2
            weights = np.ones(mu)
            # Use log weights for better performance
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1 / np.sum(weights ** 2)
            
            x_new_mean = np.zeros(d)
            for i in range(mu):
                x_new_mean += weights[i] * offspring[i]
            
            # Update covariance matrix
            # pc update
            pc = (1 - cc) * pc + cc * (x_new_mean - x_mean) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (x_new_mean - x_mean) / sigma
            
            # Rank-one and rank-mu updates
            C = (1 + c_sigma * (np.dot(ps, ps) / chi_n ** 2 - 1) * cs + 
                 cmu * np.sum(weights[:mu] * np.outer(offspring[i] - x_mean, 
                                                      offspring[i] - x_mean) / sigma ** 2 
                              for i in range(mu)) - C)
            
            # Actually, let's use the standard CMA-ES update formula
            C = (1 - c_sigma) * C + c_sigma * (np.outer(ps, ps) + 
                 (np.dot(ps, ps) / chi_n ** 2 - 1) * C)
            for i in range(mu):
                C += cmu * weights[i] * np.outer(offspring[i] - x_mean, 
                                                 offspring[i] - x_mean) / sigma ** 2
            
            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 100)
            
            x_mean = x_new_mean
            x_mean = np.clip(x_mean, -5, 5)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
