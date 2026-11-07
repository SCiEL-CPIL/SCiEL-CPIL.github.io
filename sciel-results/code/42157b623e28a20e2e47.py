import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        d = self.dim
        total_budget = self.budget
        
        best_x = np.zeros(d)
        best_f = np.inf
        calls_made = 0
        
        def safe_call(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= total_budget:
                return np.inf
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # CMA-ES implementation
        def cma_es(x0, sigma0, budget_left):
            nonlocal calls_made, best_x, best_f
            if budget_left <= 0:
                return
            
            # Population size
            # For d=5, lambda = 4 + floor(3 * ln(5)) = 4 + 4 = 8? Or 4 + 3*ln(5+d)
            # Standard: lambda = 4 + floor(3 * ln(d))
            lam = 4 + int(3 * np.log(d))
            if lam < 4: lam = 4
            
            # Step size
            sigma = sigma0
            
            # Means
            mean = x0.copy()
            best_local_x = x0.copy()
            best_local_f = safe_call(x0)
            
            # CMA parameters
            cm = np.eye(d)
            ps = np.zeros(d)
            pc = np.zeros(d)
            
            # Weights
            w = np.log(lam / 2 + 1) - np.log(np.arange(1, lam + 1))
            w = w / np.sum(w)
            mu = int(np.sum(w > 0))
            w = w[:mu]
            w = w / np.sum(w)
            
            mueff = 1 / np.sum(w)
            
            # Update rates
            ccs = (mueff + 2) / (d + mueff + 5)
            cc = (2 + mueff / d) / (d + 2 + mueff / d)
            cs = (mueff + 2) / (d + mueff + 5)
            c1 = 2 / ((d + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
            
            chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))
            
            it = 0
            stagnation_count = 0
            
            while calls_made < total_budget and time.time() < deadline:
                # Check convergence
                if sigma < 1e-10:
                    break
                
                # Generate population
                pop = []
                for i in range(lam):
                    z = np.random.randn(d)
                    x = mean + sigma * (cm @ z)
                    x = np.clip(x, -5, 5)
                    pop.append(x)
                
                # Evaluate
                vals = []
                for x in pop:
                    vals.append(safe_call(x))
                
                # Sort
                idx = np.argsort(vals)
                pop = [pop[i] for i in idx]
                vals = [vals[i] for i in idx]
                
                # Update best
                if vals[0] < best_local_f:
                    best_local_f = vals[0]
                    best_local_x = pop[0].copy()
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                    
                # Check stagnation
                if stagnation_count > 10 * d:
                    break
                
                # Update mean
                old_mean = mean.copy()
                mean = np.sum(np.array([w[i] * pop[i] for i in range(mu)]), axis=0)
                
                # Update covariance matrix
                diff = (mean - old_mean) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (cm @ (mean - old_mean))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (it + 1))) / chiN < 1.4 + 2 / (d + 1)
                
                pc = (1 - cc) * pc + hsig * cc * diff
                vpc = (pc @ pc) / d - 1
                cm = (1 + c1 * vpc + cmu * vpc - c1 - cmu) * cm + c1 * np.outer(ps, ps) + cmu * np.sum([w[i] * np.outer((pop[i] - old_mean) / sigma, (pop[i] - old_mean) / sigma) for i in range(mu)], axis=0)
                
                # Update sigma
                sigma = sigma * np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1))
                
                it += 1
                
                # Early stop if budget is tight
                if lam > budget_left - calls_made:
                    break

        # Multi-start strategy
        # Start with large step, then smaller
        starts = [
            (np.zeros(d), 4.0),
            (np.ones(d), 1.0),
            (-np.ones(d), 1.0),
            (np.random.uniform(-2, 2, d), 0.5)
        ]
        
        # Shuffle starts slightly? No, deterministic is better for reproducibility if seed is fixed, 
        # but random is fine if we want diversity. The task says "hidden seeded shift", so the function is fixed.
        # We use fixed starts to be deterministic.
        
        for x0, sigma0 in starts:
            if calls_made >= total_budget or time.time() > deadline:
                break
            cma_es(x0, sigma0, total_budget - calls_made)
            
            # If we found a very good solution, maybe stop?
            # For Sphere, if we get close to 0 (relative to offset), we are done.
            # But we don't know f_opt.
            # We can check if sigma becomes very small or if no improvement for many iterations.
            # The inner loop handles stagnation.
            
        return best_x, best_f
