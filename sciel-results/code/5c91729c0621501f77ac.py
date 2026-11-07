"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution


class Optimizer:
    """Differential Evolution based optimizer for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        # Use scipy's differential_evolution with a bounded box
        # We need to track the number of function calls to stay within budget
        # scipy's differential_evolution doesn't easily let us limit calls precisely
        # so we implement our own DE or use a wrapper.
        
        # Let's implement a simple Differential Evolution manually to control budget.
        
        rng = np.random.default_rng(42)
        
        # Population size
        pop_size = 20
        dim = self.dim
        budget = self.budget
        
        # Initialize population uniformly in [-5, 5]
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fitness = np.array([func(x) for x in pop])
        
        calls_used = pop_size
        
        # DE parameters
        F = 0.8  # scale factor
        CR = 0.9  # crossover rate
        max_iter = (budget - calls_used) // pop_size
        
        # Keep track of best
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]
        
        for gen in range(max_iter):
            new_pop = pop.copy()
            new_fitness = fitness.copy()
            
            for i in range(pop_size):
                if calls_used >= budget:
                    break
                    
                # Select two distinct random indices different from i
                r1, r2 = rng.choice(pop_size, 2, replace=False)
                while r1 == i or r2 == i:
                    r1, r2 = rng.choice(pop_size, 2, replace=False)
                
                # Mutation
                mutant = pop[r1] + F * (pop[r2] - pop[i])
                
                # Crossover
                trial = pop[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                calls_used += 1
                
                # Selection
                if f_trial <= fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
            
            pop = new_pop
            fitness = new_fitness
            
            if calls_used >= budget:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
