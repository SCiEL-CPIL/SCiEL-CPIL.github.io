"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution optimizer for Rastrigin-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Population size
        N = 15
        # Differential weight
        F = 0.8
        # Crossover rate
        CR = 0.9

        # Initialize population
        population = rng.uniform(-5, 5, (N, dim))
        fitness = np.array([func(x) for x in population])
        calls_made = N

        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_f = fitness[best_idx]

        # Adaptive parameters: decrease F as we converge
        # Simple strategy: keep F constant, but maybe adjust CR?
        # For Rastrigin, high CR is good for exploration.
        
        while calls_made < budget:
            for i in range(N):
                if calls_made >= budget:
                    break
                
                # DE/rand/1
                # Select three distinct random indices
                while True:
                    r1 = rng.integers(0, N)
                    r2 = rng.integers(0, N)
                    r3 = rng.integers(0, N)
                    if r1 != i and r2 != i and r3 != i and r1 != r2 and r2 != r3 and r1 != r3:
                        break
                
                # Create mutant vector
                mutant = population[r1] + F * (population[r2] - population[r3])
                
                # Binary Crossover
                trial = population[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                calls_made += 1
                
                # Selection
                if f_trial <= fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
