"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Differential Evolution (DE) with adaptive restarts and best-point re-seeding."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 275  # Safety margin for 300s limit
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        budget = self.budget
        dim = self.dim

        # DE Parameters
        pop_size = 20
        F = 0.8  # Scale factor
        CR = 0.9  # Crossover rate
        
        # Adaptive restart strategy:
        # We allocate budget chunks to DE runs.
        # Early runs explore broadly. Later runs exploit around the best found.
        # Total budget is 500.
        # Let's do 10 restarts.
        max_restarts = 10
        calls_per_restart = max(1, budget // max_restarts)
        
        # To maximize AOCC (Area Over Convergence Curve), we want fast initial progress.
        # DE with a decent population size converges faster than (1+1)-ES in high dims or rugged landscapes.
        
        for restart in range(max_restarts):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            
            # Initialize population
            # For the first restart, use a wide range.
            # For subsequent restarts, mix: some random, some near best_x.
            if best_x is None:
                # First restart: wide exploration
                pop = rng.uniform(-5, 5, (pop_size, dim))
            else:
                # Subsequent restarts: 50% near best, 50% random (to escape local minima)
                # The "near best" points are perturbed by a larger sigma to escape the local basin
                sigma_explore = 2.0
                near_best = np.clip(best_x + rng.normal(0, sigma_explore, (pop_size // 2, dim)), -5, 5)
                random_part = rng.uniform(-5, 5, (pop_size - pop_size // 2, dim))
                pop = np.vstack([near_best, random_part])
            
            # Evaluate initial population
            pop_f = np.array([func(x) for x in pop])
            calls_made += pop_size
            
            # Update global best
            min_idx = np.argmin(pop_f)
            if pop_f[min_idx] < best_f:
                best_f = pop_f[min_idx]
                best_x = pop[min_idx].copy()
            
            # Main DE loop
            # We run until we've used our allocated budget for this restart or hit the deadline
            while calls_made < budget and time.time() < deadline:
                # Generate trial vector
                # Select 3 distinct random individuals
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                
                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                v = pop[r1] + F * (pop[r2] - pop[r3])
                v = np.clip(v, -5, 5)
                
                # Crossover: binomial crossover
                j_rand = rng.integers(0, dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[r1][j]
                
                # Evaluate trial
                trial_f = func(trial)
                calls_made += 1
                
                # Update global best
                if trial_f < best_f:
                    best_f = trial_f
                    best_x = trial.copy()
                
                # Selection
                if trial_f <= pop_f[r1]:
                    pop[r1] = trial
                    pop_f[r1] = trial_f
                    
                # Check for stagnation? 
                # DE is robust, but if we haven't improved best_f in a long time within this restart,
                # we might break early to allow more restarts.
                # However, DE's diversity usually handles this. 
                # Let's just stick to the budget allocation for simplicity and reliability.
                
            # If we found a very good solution, stop early
            if best_f < -80:
                break
                
        return best_x, best_f
# EVOLVE-BLOCK-END
