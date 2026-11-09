"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution optimizer for Rastrigin-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Population size
        pop_size = 10
        # Ensure we have at least pop_size + 1 evaluations to start
        if budget < pop_size + 1:
            # Fallback to random search if budget is tiny
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            return x, f

        # Initialize population
        pop = rng.uniform(-5, 5, (pop_size, dim))
        pop_f = np.array([func(x) for x in pop])
        
        calls = pop_size
        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]
        
        # DE parameters
        F = 0.8
        CR = 0.9
        
        while calls < budget:
            # For each individual in population
            for i in range(pop_size):
                if calls >= budget:
                    break
                
                # Select 3 distinct random indices different from i
                r1, r2, r3 = i
                while r1 == i:
                    r1 = rng.integers(0, pop_size)
                while r2 == i or r2 == r1:
                    r2 = rng.integers(0, pop_size)
                while r3 == i or r3 == r1 or r3 == r2:
                    r3 = rng.integers(0, pop_size)
                    
                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                v = pop[r1] + F * (pop[r2] - pop[r3])
                v = np.clip(v, -5, 5)
                
                # Crossover
                trial = pop[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                trial = np.clip(trial, -5, 5)
                
                # Evaluate trial
                f_trial = func(trial)
                calls += 1
                
                # Selection
                if f_trial <= pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        
        return best_x, best_f
# EVOLVE-BLOCK-END
