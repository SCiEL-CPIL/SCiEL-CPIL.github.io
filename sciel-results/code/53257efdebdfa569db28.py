"""Rewritten optimizer using Differential Evolution (DE) for BBOB-style f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution optimizer for BBOB-style f4 (Büche-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # DE Parameters
        # Population size: 20 is a good balance for dim=5 and budget=500 (25 generations)
        npop = 20
        F = 0.8  # Mutation factor
        CR = 0.5  # Crossover rate
        
        # Initialize population
        pop = rng.uniform(-5, 5, (npop, dim))
        fitness = np.array([func(x) for x in pop])
        
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]
        
        evals_used = npop
        
        # Main loop
        while evals_used + npop <= budget:
            # Generate trial vectors
            trial_pop = np.zeros_like(pop)
            trial_fitness = np.zeros(npop)
            
            for i in range(npop):
                # Select two distinct random indices r1, r2 different from i
                # Using DE/best/1 strategy: x_best + F * (x_r1 - x_r2)
                r1, r2 = rng.choice(np.delete(np.arange(npop), i), size=2, replace=False)
                
                # Mutation
                v = pop[best_idx] + F * (pop[r1] - pop[r2])
                
                # Crossover (Binomial)
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial_pop[i, j] = v[j]
                    else:
                        trial_pop[i, j] = pop[i, j]
                
                # Clip to bounds
                trial_pop[i] = np.clip(trial_pop[i], -5, 5)
                
                # Evaluate
                trial_fitness[i] = func(trial_pop[i])
                evals_used += 1
                
                # Selection
                if trial_fitness[i] <= fitness[i]:
                    pop[i] = trial_pop[i]
                    fitness[i] = trial_fitness[i]
                    
            # Update global best
            new_best_idx = np.argmin(fitness)
            if fitness[new_best_idx] < best_f:
                best_f = fitness[new_best_idx]
                best_x = pop[new_best_idx].copy()
                best_idx = new_best_idx
                
        return best_x, best_f
# EVOLVE-BLOCK-END
