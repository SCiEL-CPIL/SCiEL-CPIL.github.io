"""
Differential Evolution (DE) optimizer for BBOB f7 (Step ellipsoidal) in 5D.

Strategy: DE with adaptive parameters and population diversity maintenance.
DE is a population-based evolutionary algorithm that uses vector differences
between solutions to generate mutations. It is robust to non-smooth functions
like the step ellipsoidal and can explore the search space effectively.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer using Differential Evolution with adaptive parameters.
    
    DE is well-suited for non-smooth, multi-modal functions like the step
    ellipsoidal. It maintains population diversity through mutation and
    crossover, and adapts its parameters during the search.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        # Population size for DE
        pop_size = max(4 * dim, 10)  # Standard recommendation: 4*dim to 10*dim
        pop_size = min(pop_size, 50)  # Cap to stay within budget
        
        # DE parameters (adaptive)
        F = 0.8  # Scale factor for mutation
        CR = 0.9  # Crossover rate
        
        # Initialize population uniformly in [-5, 5]^dim
        pop = np.random.uniform(-5, 5, size=(pop_size, dim))
        
        # Evaluate initial population
        fitness = np.empty(pop_size)
        calls_made = 0
        for i in range(pop_size):
            if calls_made >= budget:
                break
            fitness[i] = func(pop[i])
            calls_made += 1
        
        # Track best
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]
        
        # Main loop
        gen = 0
        no_improve = 0
        max_no_improve = 50  # Restart after this many generations without improvement
        
        while calls_made < budget:
            # Check time
            if time.time() > deadline:
                break
            
            gen += 1
            prev_best_f = best_f
            
            # Adaptive parameter adjustment
            # Reduce F slightly as we progress to fine-tune
            progress = min(1.0, calls_made / (budget * 0.8))
            F_current = 0.8 - 0.3 * progress
            CR_current = 0.9 - 0.2 * progress
            
            # Generate offspring
            offspring = np.empty_like(pop)
            offspring_fitness = np.empty(pop_size)
            
            for i in range(pop_size):
                if calls_made >= budget:
                    break
                
                # Select three distinct individuals for mutation
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                
                # DE/rand/1 mutation: v = x_r1 + F * (x_r2 - x_r3)
                v = pop[r1] + F_current * (pop[r2] - pop[r3])
                
                # Binary crossover
                trial = pop[i].copy()
                jrand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR_current or j == jrand:
                        trial[j] = v[j]
                
                # Clip to box
                trial = np.clip(trial, -5, 5)
                
                # Evaluate trial
                trial_f = func(trial)
                calls_made += 1
                offspring[i] = trial
                offspring_fitness[i] = trial_f
                
                # Selection
                if trial_f < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_f
                    if trial_f < best_f:
                        best_f = trial_f
                        best_x = trial.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            
            if calls_made >= budget:
                break
            
            # Check for stagnation
            if no_improve >= max_no_improve:
                # Restart: reset population around the best solution
                # Keep best solution, regenerate others
                for i in range(pop_size):
                    if i == np.argmin(fitness):
                        continue  # Keep the best
                    pop[i] = best_x + np.random.normal(0, 0.5, size=dim)
                    pop[i] = np.clip(pop[i], -5, 5)
                    if calls_made >= budget:
                        break
                    fitness[i] = func(pop[i])
                    calls_made += 1
                no_improve = 0
                # Reduce exploration by lowering F
                F = max(0.3, F * 0.8)
            
            # Stop if convergence is very tight
            if np.max(fitness) - np.min(fitness) < 1e-10:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
