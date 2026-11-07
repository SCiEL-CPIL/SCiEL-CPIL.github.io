"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Population-based (mu+lambda)-ES for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Deadline guard: evaluator kills after 330s. 
        # We allow 300s for safety.
        start_time = time.time()
        max_time = 300.0
        
        rng = np.random.default_rng(42)
        
        # Parameters for (mu+lambda)-ES
        # For dim=5, a population size of 10-20 is reasonable.
        # mu is the number of parents, lambda is offspring count.
        # Let's use mu=5, lambda=20.
        mu = 5
        lam = 20
        
        # Initial population: random points in [-5, 5]^dim
        # We need to evaluate them to start.
        pop = rng.uniform(-5, 5, size=(lam, dim))
        # Clip to be safe, though uniform is already in bounds
        pop = np.clip(pop, -5, 5)
        
        # Evaluate initial population
        fitness = np.array([func(x) for x in pop])
        
        # Sort indices by fitness (ascending)
        # We keep the best mu
        idx = np.argsort(fitness)
        parents = pop[idx[:mu]]
        parent_fitness = fitness[idx[:mu]]
        
        # Mean of parents
        x_mean = np.mean(parents, axis=0)
        
        # Initial step size
        # Heuristic: based on box size
        sigma = 2.0
        
        # CMA-ES like covariance matrix adaptation? 
        # For simplicity and robustness in 5D, a standard (mu+lambda)-ES with 
        # adaptive sigma might be sufficient, but CMA-ES is better for rotated landscapes.
        # However, implementing full CMA-ES is complex. 
        # Let's try a simple (mu+lambda)-ES with covariance matrix adaptation 
        # or just standard ES with a larger population.
        # Given the "Lunacek bi-Rastrigin" is multimodal and potentially rotated/shifted,
        # diversity is key.
        
        # Let's use a simple (mu+lambda)-ES with re-evaluation of the mean?
        # Or just standard (mu+lambda)-ES.
        
        # To handle rotation, CMA-ES is superior. But standard ES with large lambda 
        # and small mu can also work if the landscape isn't too ill-conditioned.
        # Let's stick to a robust (mu+lambda)-ES first.
        
        # Adaptive sigma: 1/5 rule is for (1+1). For (mu+lambda), we can use 
        # a similar heuristic or just fixed sigma with restarts?
        # Let's try adaptive sigma based on the number of successful steps.
        
        # State variables
        best_x = parents[0]
        best_f = parent_fitness[0]
        
        # We will perform generations until budget is exhausted or time runs out.
        # Each generation costs 'lam' evaluations.
        # Initial cost: lam
        # Remaining budget: budget - lam
        
        # We need to be careful with the budget count.
        # The problem says "More than 500 calls... makes the whole result invalid".
        # So we must not exceed 500 calls.
        
        calls_made = lam
        remaining = budget - calls_made
        
        if remaining <= 0:
            return best_x, best_f
            
        # Main loop
        while remaining >= lam:
            if time.time() - start_time > max_time - 5:
                break
                
            # Generate offspring
            # Sample from N(x_mean, sigma^2 * I)
            # Simple ES assumes spherical covariance.
            # If the landscape is rotated, this is inefficient.
            # But without CMA, this is the baseline.
            
            offspring = x_mean + sigma * rng.standard_normal((lam, dim))
            offspring = np.clip(offspring, -5, 5)
            
            # Evaluate offspring
            off_fitness = np.array([func(x) for x in offspring])
            calls_made += lam
            remaining -= lam
            
            # Combine parents and offspring
            # For (mu+lambda) selection: select best mu from all (mu+lambda)
            # But we don't have the old parents' fitness in the same array if we discard them?
            # In (mu+lambda), we generate lam offspring, combine with mu parents, select mu.
            # We have `parents` (mu) and `offspring` (lam).
            
            all_pop = np.vstack([parents, offspring])
            all_fit = np.concatenate([parent_fitness, off_fitness])
            
            # Update global best
            min_idx = np.argmin(all_fit)
            if all_fit[min_idx] < best_f:
                best_f = all_fit[min_idx]
                best_x = all_pop[min_idx]
                
            # Select new parents
            idx = np.argsort(all_fit)
            parents = all_pop[idx[:mu]]
            parent_fitness = all_fit[idx[:mu]]
            
            # Update mean
            x_mean = np.mean(parents, axis=0)
            
            # Adapt sigma
            # Simple heuristic: if best improved, increase sigma slightly?
            # Or use 1/5 rule logic?
            # Let's keep sigma fixed or slowly decrease it to exploit.
            # For multimodal, fixed sigma might get stuck.
            # Let's try a simple adaptation:
            # Count how many offspring were better than the worst parent?
            # Or just use a fixed sigma that is large enough to explore.
            # Given dim=5 and box [-5,5], sigma=2.0 is reasonable for exploration.
            # Maybe decrease it slowly as we get closer to optimum?
            # We don't know the optimum.
            
            # Let's try a simple restart mechanism or just keep sigma constant.
            # Constant sigma is safer for multimodal.
            pass
            
        return best_x, best_f
# EVOLVE-BLOCK-END
