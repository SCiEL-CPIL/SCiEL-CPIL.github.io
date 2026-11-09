"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution optimizer for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Parameters
        pop_size = 10
        F = 0.8  # Scale factor
        CR = 0.9  # Crossover rate
        
        # Initialize population
        # Uniform distribution in [-5, 5]
        pop = np.random.uniform(-5, 5, (pop_size, dim))
        fitness = np.array([func(x) for x in pop])
        calls_used = pop_size
        
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]
        
        # Main loop
        while calls_used < budget:
            for i in range(pop_size):
                if calls_used >= budget:
                    break
                
                # Select three distinct random indices different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)
                
                # Mutation: v = x_a + F * (x_b - x_c)
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Clip to bounds
                v = np.clip(v, -5, 5)
                
                # Crossover: binomial
                # Create a random vector to decide which components to take from v
                r = np.random.rand(dim)
                j_rand = np.random.randint(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if r[j] < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Evaluate trial
                f_trial = func(trial)
                calls_used += 1
                
                # Selection
                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
