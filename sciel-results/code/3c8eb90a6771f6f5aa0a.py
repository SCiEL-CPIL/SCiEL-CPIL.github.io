"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """(mu + lambda)-ES with population-based exploration and adaptive step size."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Deadline guard
        start_time = time.time()
        max_time = 280.0  # Safety margin
        
        # Parameters for (mu + lambda)-ES
        # For dim=5, a population size of 10-15 is reasonable.
        # mu = 5 (top 5 survivors), lambda = 10 (10 offspring per generation)
        # This provides better exploration than (1+1) and can handle rotated landscapes better.
        
        mu = 5
        lam = 10
        
        # Initial population: random points in the domain
        # To ensure diversity, we can initialize around the center or randomly.
        # Random initialization is safer for unknown shifts.
        rng = np.random.default_rng(42)
        
        # Initialize population with random points in [-5, 5]^dim
        pop = rng.uniform(-5, 5, size=(lam, dim))
        fvals = np.array([func(x) for x in pop])
        calls = lam
        
        # Sort population by fitness (ascending for minimization)
        order = np.argsort(fvals)
        pop = pop[order]
        fvals = fvals[order]
        
        # Initial mean and step size
        # Mean is the best point initially, but we can use the centroid of the top mu.
        # Standard ES uses the mean of the top mu.
        mean = np.mean(pop[:mu], axis=0)
        
        # Initial step size
        # Heuristic: sigma ~ 0.5 * (max - min) / sqrt(dim) or similar.
        # Domain width is 10. Let's start with sigma = 1.0.
        sigma = 1.0
        
        # 1/5 rule for step size adaptation
        # We track successes in a window.
        # For population-based, we can count how many offspring improved upon the current best or mean.
        # A simpler approach: if the best offspring is better than the best parent, increase sigma slightly?
        # Or use a more robust rule.
        # Let's use a simple rule: if the best fitness improves significantly, increase sigma.
        # Or use the standard 1/5 rule on the population level.
        # Let's track the number of offspring that are better than the current mean.
        
        window_size = 10
        successes = 0
        trials = 0
        
        best_x = pop[0].copy()
        best_f = fvals[0]
        
        while calls < budget:
            if time.time() - start_time > max_time:
                break
            
            # Generate offspring
            # Offspring = mean + sigma * N(0, I)
            # We generate lam offspring
            offspring = mean + sigma * rng.standard_normal((lam, dim))
            offspring = np.clip(offspring, -5, 5)
            
            # Evaluate offspring
            off_fvals = np.array([func(x) for x in offspring])
            calls += lam
            
            # Update global best
            if off_fvals.min() < best_f:
                best_f = off_fvals.min()
                best_x = offspring[np.argmin(off_fvals)].copy()
            
            # Combine parents and offspring
            # Parents are the current top mu from the previous generation
            # Actually, in (mu+lambda), we select the best mu from the combined set of mu parents and lam offspring.
            # But our current 'pop' has lam individuals. The top mu are the parents.
            # So we combine pop[:mu] and offspring.
            
            combined = np.vstack([pop[:mu], offspring])
            combined_fvals = np.concatenate([fvals[:mu], off_fvals])
            
            # Select top mu
            order = np.argsort(combined_fvals)
            new_pop = combined[order[:mu]]
            new_fvals = combined_fvals[order[:mu]]
            
            # Update mean
            new_mean = np.mean(new_pop, axis=0)
            
            # Adapt step size
            # Success if any offspring is better than the current best (or mean?)
            # Let's say success if the best offspring is better than the best parent.
            # best_parent_f = fvals[0]
            # best_offspring_f = off_fvals.min()
            # if best_offspring_f < best_parent_f: success
            
            # A more standard 1/5 rule for (mu+lambda) counts the fraction of offspring that are better than the mean.
            # Or simply: if the best fitness improved, increase sigma.
            # Let's use: if best_f (global) improved in this generation, count as success.
            # This might be too sparse.
            
            # Alternative: count how many offspring are better than the previous best.
            # prev_best = best_f
            # num_success = np.sum(off_fvals < prev_best)
            
            # Let's use a simpler heuristic:
            # If the new mean is closer to the global best than the old mean, increase sigma? No.
            
            # Let's stick to the 1/5 rule on the improvement of the best value.
            # If the best value improved, it's a success.
            # But this is binary per generation.
            
            # Let's try: if the best offspring is better than the worst parent, it's a success?
            # Or: if the best offspring is better than the best parent.
            
            # Let's use: success if best_offspring < best_parent
            best_parent_f = fvals[0]
            best_offspring_f = off_fvals.min()
            
            if best_offspring_f < best_parent_f:
                successes += 1
            trials += 1
            
            if trials >= window_size:
                success_rate = successes / trials
                if success_rate > 0.2:
                    sigma *= 1.1
                elif success_rate < 0.2:
                    sigma /= 1.1
                successes = 0
                trials = 0
            
            # Update state for next generation
            pop = new_pop
            fvals = new_fvals
            mean = new_mean
            
            # Early termination if convergence is very slow or stuck?
            # If sigma becomes too small, we might be stuck.
            # If sigma becomes too large, we are exploring widely.
            
            # If best_f hasn't changed in many generations, we might need to restart or increase sigma more aggressively.
            # For now, the 1/5 rule should handle it.
            
        return best_x, best_f

# EVOLVE-BLOCK-END
