"""Rewritten optimizer for BBOB f4 (Büche-Rastrigin) using Differential Evolution."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution based optimizer for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        rng = np.random.default_rng(42)
        
        calls_counter = [0]
        n = self.dim
        
        # DE Parameters
        pop_size = 50  # As suggested by knowledge base
        F = 0.8  # Mutation factor
        CR = 0.5  # Crossover rate
        
        # Initialize population
        population = rng.uniform(-5, 5, (pop_size, n))
        fitnesses = np.empty(pop_size)
        
        # Evaluate initial population
        for i in range(pop_size):
            if calls_counter[0] >= self.budget or time.time() > deadline:
                break
            x = population[i]
            f = func(x)
            fitnesses[i] = f
            calls_counter[0] += 1
            
        # Find initial best
        best_idx = np.argmin(fitnesses)
        best_x = population[best_idx].copy()
        best_f = fitnesses[best_idx]
        
        # DE Loop
        while calls_counter[0] < self.budget and time.time() < deadline:
            # Check if we have enough budget for a full generation
            if calls_counter[0] + pop_size > self.budget:
                # If not enough for full generation, just use remaining budget for random perturbation of best
                remaining = self.budget - calls_counter[0]
                for _ in range(remaining):
                    if time.time() > deadline:
                        break
                    # Simple random restart/perturbation
                    x_new = best_x + rng.normal(0, 0.1, n)
                    x_new = np.clip(x_new, -5, 5)
                    f_new = func(x_new)
                    calls_counter[0] += 1
                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_new.copy()
                break
                
            # Generate offspring for each individual
            for i in range(pop_size):
                if calls_counter[0] >= self.budget or time.time() > deadline:
                    break
                
                # Select 3 distinct random individuals
                idxs = rng.choice(pop_size, 3, replace=False)
                a, b, c = population[idxs[0]], population[idxs[1]], population[idxs[2]]
                
                # Mutation
                v = a + F * (b - c)
                
                # Crossover
                u = rng.random(n) < CR
                trial = np.where(u, v, population[i])
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                calls_counter[0] += 1
                
                # Selection
                if f_trial <= fitnesses[i]:
                    population[i] = trial
                    fitnesses[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
