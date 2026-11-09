"""
Adaptive Step-Size Evolution Strategy (ES) Optimizer
Strategy:
- Uses a population-based ES with lambda=10 offspring per generation.
- Employs cumulative step-size adaptation (CSA) to dynamically adjust sigma.
- Avoids the computational overhead of CMA-ES covariance matrix updates.
- Starts at the origin, leveraging the known locality of f14's minimum.
- Designed to maximize early progress (AOCC) by converging quickly in initial steps.
"""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        calls_used = 0
        rng = np.random.default_rng(42)

        def check_time():
            nonlocal calls_used
            return (time.time() >= deadline) or (calls_used >= budget)

        def evaluate(x):
            nonlocal calls_used
            # The problem statement says func clips, but we can clip here too for safety if func doesn't
            # However, the contract says "func(x) clips... and returns a float". 
            # We should pass x as is, or clip it? 
            # "func(x) clips a finite length-5 vector to the box". So func handles clipping.
            # But if we generate points outside, it's fine.
            f = func(x)
            calls_used += 1
            return f

        # ES Parameters
        lambda_pop = 10
        mu = lambda_pop // 2  # 5 parents
        
        # Weights for parent selection
        # Standard weights: w_i = log(mu + 0.5) - log(i)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Initial state
        mean = np.zeros(dim)
        sigma = 1.0
        
        # Step size control parameters
        # c_sigma and d_sigma for cumulative step size adaptation
        # c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        # d_sigma = 1 + 2 * max(0, sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Cumulative step size vector
        p_sigma = np.zeros(dim)
        
        x_best = mean.copy()
        f_best = np.inf

        while not check_time():
            # Generate offspring
            offspring = []
            f_offspring = []
            
            for i in range(lambda_pop):
                if check_time():
                    break
                z = rng.standard_normal(dim)
                x = mean + sigma * z
                offspring.append(x)
                f = evaluate(x)
                f_offspring.append(f)
                
                if f < f_best:
                    f_best = f
                    x_best = x.copy()
            
            if len(f_offspring) < mu:
                break
            
            # Sort offspring
            indices = np.argsort(f_offspring)
            offspring = [offspring[i] for i in indices]
            f_offspring = [f_offspring[i] for i in indices]
            
            # Update mean
            old_mean = mean.copy()
            mean = np.sum([weights[i] * offspring[i] for i in range(mu)], axis=0)
            
            # Update step size (sigma) using cumulative step size adaptation
            # The standard update for sigma in simple ES with CSA is:
            # sigma *= exp( (c_sigma / d_sigma) * ( ||p_sigma|| / sqrt(mu_eff) - 1 ) )
            # where p_sigma is updated as:
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (mean - old_mean) / sigma
            
            # Note: In full CMA-ES, p_sigma is whitened by C^-1/2. Here C=I initially, 
            # and we are not updating C, so we assume the search is isotropic.
            # This is a valid strategy: "Adaptive Step-Size ES".
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * ((mean - old_mean) / sigma)
            
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma *= np.exp((c_sigma / d_sigma) * (norm_p_sigma / np.sqrt(mu_eff) - 1))
            
            # Prevent sigma from becoming too small or too large
            sigma = np.clip(sigma, 1e-6, 10.0)

        return x_best, f_best
