"""Rewritten program: Differential Evolution optimizer for BBOB f3 (Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution continuous optimizer for Rastrigin-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        # DE Parameters
        # Population size
        NP = 10
        # Mutation factor
        F = 0.8
        # Crossover rate
        CR = 0.9
        
        # Bounds
        lb = -5.0
        ub = 5.0
        
        # Initialize population uniformly in [lb, ub]
        # x is (NP, n)
        pop = rng.uniform(lb, ub, size=(NP, n))
        
        # Evaluate initial population
        fitness = np.array([func(pop[i]) for i in range(NP)])
        
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]
        
        calls_made = NP
        
        # Main loop
        while calls_made < budget:
            # Determine how many trials we can make in this generation
            remaining = budget - calls_made
            if remaining < NP:
                num_trials = remaining
            else:
                num_trials = NP
            
            for i in range(num_trials):
                # Choose 3 distinct random indices
                r1, r2, r3 = rng.choice(NP, size=3, replace=False)
                
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Clip to bounds
                v = np.clip(v, lb, ub)
                
                # Crossover
                u = np.copy(pop[i])
                j_rand = rng.integers(n)
                
                for j in range(n):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                
                # Evaluate trial vector
                f_u = func(u)
                calls_made += 1
                
                # Selection
                if f_u <= fitness[i]:
                    pop[i] = u
                    fitness[i] = f_u
                    
                    if f_u < best_f:
                        best_f = f_u
                        best_x = u.copy()
                        
            # If we exhausted the budget, break out of the while loop
            if calls_made >= budget:
                break
                
        return best_x, best_f
# EVOLVE-BLOCK-END
