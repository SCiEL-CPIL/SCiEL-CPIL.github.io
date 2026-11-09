"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution optimizer for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Differential Evolution Parameters
        # Population size: typically 5-10 * dim
        # With dim=5, budget=500, we can afford a larger population for better exploration
        npop = 15 
        F = 0.8  # Scale factor
        CR = 0.5 # Crossover rate
        
        # Initialize population
        pop = rng.uniform(-5, 5, size=(npop, dim))
        fits = np.zeros(npop)
        calls_made = 0
        
        for i in range(npop):
            fits[i] = func(pop[i])
            calls_made += 1
            
        best_idx = np.argmin(fits)
        best_x = pop[best_idx].copy()
        best_f = fits[best_idx]
        
        # Main loop
        while calls_made < budget:
            new_pop = np.zeros_like(pop)
            new_fits = np.zeros_like(fits)
            
            for i in range(npop):
                if calls_made >= budget:
                    break
                    
                # Select three distinct random individuals
                r1, r2, r3 = rng.choice(npop, size=3, replace=False)
                while r1 == i or r2 == i or r3 == i or r1 == r2 or r2 == r3 or r1 == r3:
                    r1, r2, r3 = rng.choice(npop, size=3, replace=False)
                
                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover: binary
                cross_mask = rng.random(dim) < CR
                # Ensure at least one gene is crossed
                if not np.any(cross_mask):
                    j = rng.randint(dim)
                    cross_mask[j] = True
                    
                trial = np.where(cross_mask, v, pop[i])
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                calls_made += 1
                
                # Selection
                if f_trial <= fits[i]:
                    new_pop[i] = trial
                    new_fits[i] = f_trial
                else:
                    new_pop[i] = pop[i]
                    new_fits[i] = fits[i]
                    
            # Update population
            pop = new_pop
            fits = new_fits
            
            # Track best
            curr_best_idx = np.argmin(fits)
            if fits[curr_best_idx] < best_f:
                best_f = fits[curr_best_idx]
                best_x = pop[curr_best_idx].copy()
                
        return best_x, best_f
# EVOLVE-BLOCK-END
