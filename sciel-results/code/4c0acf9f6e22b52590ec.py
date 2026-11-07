"""Rewritten program: Differential Evolution optimizer."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution style optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        # Population size for DE
        # Typically 5-10 * dim is good. For d=5, 25-50.
        # Budget is 500. If we use 25 individuals, we can do ~20 generations.
        # Let's aim for a balance. 
        pop_size = 25
        
        # Initialize population uniformly in [-5, 5]
        population = rng.uniform(-5, 5, (pop_size, d))
        
        # Evaluate initial population
        fitnesses = np.zeros(pop_size)
        budget_used = 0
        for i in range(pop_size):
            if budget_used >= budget:
                break
            fitnesses[i] = func(population[i])
            budget_used += 1
        
        # Find initial best
        best_idx = np.argmin(fitnesses)
        global_best_x = population[best_idx].copy()
        global_best_f = fitnesses[best_idx]

        # DE Parameters
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate

        # Main loop
        while budget_used < budget:
            new_population = []
            new_fitnesses = []
            
            # Generate offspring
            for i in range(pop_size):
                if budget_used >= budget:
                    break
                
                # Select three distinct random indices
                r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                # Ensure they are not i
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                
                # Mutation: v_i = x_r1 + F * (x_r2 - x_r3)
                v = population[r1] + F * (population[r2] - population[r3])
                
                # Crossover: Binary crossover
                # Generate random j0
                j0 = rng.integers(0, d)
                cross_mask = rng.random(d) < CR
                cross_mask[j0] = True # Ensure at least one gene is from v
                
                # Create trial vector
                trial = np.where(cross_mask, v, population[i])
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                budget_used += 1
                
                # Selection
                if f_trial <= fitnesses[i]:
                    new_population.append(trial)
                    new_fitnesses.append(f_trial)
                    # Update global best
                    if f_trial < global_best_f:
                        global_best_f = f_trial
                        global_best_x = trial.copy()
                else:
                    new_population.append(population[i])
                    new_fitnesses.append(fitnesses[i])
            
            if len(new_population) < pop_size:
                break
                
            population = np.array(new_population)
            fitnesses = np.array(new_fitnesses)
            
            # Optional: Restart if stuck? 
            # For now, simple DE.
            
            # Check for convergence
            if np.std(fitnesses) < 1e-8:
                break

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
