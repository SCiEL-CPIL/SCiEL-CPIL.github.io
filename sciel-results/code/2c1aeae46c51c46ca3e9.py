"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # DE Parameters
        NP = 10  # Population size
        F = 0.8  # Scale factor
        CR = 0.5 # Crossover rate
        
        # Initialize population
        pop = rng.uniform(-5, 5, size=(NP, dim))
        
        # Evaluate initial population
        fitness = np.empty(NP)
        evals_used = 0
        for i in range(NP):
            fitness[i] = func(pop[i])
            evals_used += 1
            
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]
        
        # Main Loop
        # We ensure we don't exceed budget by checking if a full generation fits
        while evals_used + NP <= budget:
            # Generate new population
            for i in range(NP):
                # Select 3 distinct random indices different from i
                candidates = [j for j in range(NP) if j != i]
                r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
                
                # Mutation
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                trial = pop[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                        
                # Bounds check
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                evals_used += 1
                
                # Selection
                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        
        return best_x, best_f

# EVOLVE-BLOCK-END
