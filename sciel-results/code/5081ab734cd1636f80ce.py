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
        
        # Population size: typically 5-10 * dim
        pop_size = max(10, 5 * dim)
        
        # Initialize population uniformly in [-5, 5]
        population = rng.uniform(-5, 5, (pop_size, dim))
        fitness = np.array([func(x) for x in population])
        calls_made = pop_size
        
        # DE parameters
        F = 0.8  # Scale factor
        CR = 0.9  # Crossover rate
        
        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_f = fitness[best_idx]
        
        # Run DE until budget is exhausted
        while calls_made < budget:
            # Update best
            best_idx = np.argmin(fitness)
            best_f = fitness[best_idx]
            best_x = population[best_idx].copy()
            
            # Generate new individuals
            for i in range(pop_size):
                if calls_made >= budget:
                    break
                
                # Select three distinct random indices different from i
                r1, r2, r3 = i
                while r1 == i:
                    r1 = rng.integers(0, pop_size)
                while r2 == i or r2 == r1:
                    r2 = rng.integers(0, pop_size)
                while r3 == i or r3 == r1 or r3 == r2:
                    r3 = rng.integers(0, pop_size)
                
                # Differential evolution mutation
                v = population[r1] + F * (population[r2] - population[r3])
                
                # Binary crossover
                trial = np.copy(population[i])
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate trial
                f_trial = func(trial)
                calls_made += 1
                
                # Selection
                if f_trial <= fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
        
        # Final best
        best_idx = np.argmin(fitness)
        best_f = fitness[best_idx]
        best_x = population[best_idx].copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
