import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        start_time = time.time()
        time_limit = 325.0 

        x_best = None
        f_best = np.inf
        eval_count = 0
        
        def evaluate(x):
            nonlocal eval_count, f_best, x_best
            if eval_count >= budget:
                return None
            if time.time() - start_time > time_limit:
                return None
            try:
                fx = func(x)
            except Exception:
                return None
            eval_count += 1
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            return fx

        # CMA-ES Parameters
        lambda_ = 4  # Small population for tight budget
        mu = max(1, lambda_ // 2)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt(mu_eff / (n + 2) - 1)) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        def run_cmaes(start_x, start_sigma):
            nonlocal eval_count, f_best, x_best
            mean = start_x.copy()
            sigma = start_sigma
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            # Pre-compute inverse of sqrt(C) for p_sigma update? 
            # We will compute it on the fly.
            
            gen = 0
            stagnation = 0
            restart_threshold = 10 # Generations
            
            while eval_count < budget:
                if time.time() - start_time > time_limit:
                    break
                
                # Generate population
                offsprings = []
                for _ in range(lambda_):
                    try:
                        L = np.linalg.cholesky(C)
                        z = np.random.standard_normal(n)
                        y = L @ z
                    except np.linalg.LinAlgError:
                        # Fallback if C is not positive definite
                        y = np.random.standard_normal(n)
                    x_i = mean + sigma * y
                    offsprings.append(x_i)
                    
                # Evaluate
                f_ys = []
                for x_i in offsprings:
                    f_i = evaluate(x_i)
                    if f_i is None:
                        return
                    f_ys.append(f_i)
                    
                if len(f_ys) < lambda_:
                    break
                    
                # Sort
                idx = np.argsort(f_ys)
                ys_sorted = [offsprings[i] for i in idx]
                f_ys_sorted = [f_ys[i] for i in idx]
                
                # Update mean
                new_mean = np.zeros(n)
                for i in range(mu):
                    new_mean += weights[i] * ys_sorted[i]
                    
                # Update p_sigma
                try:
                    L_inv = np.linalg.inv(np.linalg.cholesky(C))
                except np.linalg.LinAlgError:
                    L_inv = np.eye(n)
                    
                p_sigma = (1 - c_sigma) * ps + c_sigma * np.sqrt(mu_eff) * (L_inv @ (new_mean - mean)) / sigma
                ps = p_sigma
                
                # Adapt sigma
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * np.sqrt(mu_eff + 1 - mu_eff):
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
                    
                # Update C
                pc = (1 - c_c) * pc + c_c * (new_mean - mean) / sigma
                delta_heav = 1 + 2 * max(0, (np.linalg.norm(pc)**2 / (1 - (1 - c_c)**(2 * (gen + 1))) - n) / n)
                
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    diff = (ys_sorted[i] - mean) / sigma
                    rank_mu += weights[i] * np.outer(diff, diff)
                    
                C = (1 + c_1 * (delta_heav - 1) - c_1 - c_mu) * C + c_1 * np.outer(pc, pc) + c_mu * rank_mu
                C = (C + C.T) / 2
                
                mean = new_mean
                gen += 1
                
                # Check stagnation
                if f_ys_sorted[0] < f_best - 1e-8:
                    stagnation = 0
                else:
                    stagnation += 1
                    
                if stagnation > restart_threshold:
                    break
                    
            return mean, sigma

        # Multi-Start Strategy
        # Start with a random point
        x_start = np.random.uniform(-5, 5, n)
        f_start = evaluate(x_start)
        if f_start is None:
            return x_best, f_best
        
        # Run CMA-ES
        while eval_count < budget:
            if time.time() - start_time > time_limit:
                break
            
            # Run CMA-ES from current best or random
            if np.random.random() < 0.3 and eval_count > 50:
                x_start = np.random.uniform(-5, 5, n)
                sigma_start = 2.0
            else:
                x_start = x_best.copy()
                sigma_start = 1.0
                
            run_cmaes(x_start, sigma_start)
            
            # If we still have budget, we loop. 
            # The inner loop breaks on stagnation or budget/time.
            # If it broke on stagnation, we restart from a new point or best.
            # If it broke on budget/time, the outer loop will exit.
            
            # To prevent infinite loop if run_cmaes returns immediately due to time,
            # we check time again.
            if time.time() - start_time > time_limit:
                break

        return x_best, f_best
