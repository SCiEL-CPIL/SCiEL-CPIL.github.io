"""Differential Evolution optimizer for BBOB-style Rastrigin f3."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer using Differential Evolution (DE).
    DE is a robust global optimization algorithm suitable for multimodal functions
    like Rastrigin, avoiding the local minima traps of local search methods
    like Nelder-Mead.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin
        rng = np.random.default_rng(42)

        dim = self.dim
        n = self.budget
        x_best = None
        f_best = np.inf
        calls_used = 0

        def check_deadline():
            return time.time() > deadline

        def update_best(x, fx):
            nonlocal x_best, f_best
            if fx < f_best:
                f_best = fx
                x_best = x.copy()

        # DE Parameters
        # Population size: at least 4*dim is recommended, but 15 is a good balance
        # for dim=5 and budget=500.
        pop_size = min(15, n)
        if pop_size < 4:
            # Fallback to random search if budget is tiny
            for i in range(n):
                if check_deadline(): break
                x = rng.uniform(-5, 5, dim)
                fx = func(x)
                calls_used += 1
                update_best(x, fx)
            return x_best, f_best

        F = 0.8  # Mutation scale factor
        CR = 0.9  # Crossover rate

        # 1. Initialize Population
        population = []
        fitness = []
        
        for i in range(pop_size):
            if calls_used >= n or check_deadline():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_used += 1
            population.append(x)
            fitness.append(fx)
            update_best(x, fx)
            
        if not population:
            return x_best, f_best

        # 2. Main DE Loop (Greedy Update)
        while calls_used < n and not check_deadline():
            # Generate offspring for each individual in the current population
            # We iterate through indices 0 to len(population)-1
            for i in range(len(population)):
                if calls_used >= n or check_deadline():
                    break
                
                # Select base vector and two distinct random vectors for DE/rand/1
                # We need 3 distinct indices: i, r1, r2
                # If population is small, handle carefully
                if len(population) < 4:
                    # Fallback: random perturbation
                    mutant = population[i] + F * rng.standard_normal(dim)
                else:
                    # Pick r1 and r2 distinct from i and each other
                    candidates = [j for j in range(len(population)) if j != i]
                    r1 = int(rng.choice(candidates))
                    candidates.remove(r1)
                    r2 = int(rng.choice(candidates))
                    
                    # Mutation: x_r1 + F * (x_r2 - x_i)
                    mutant = population[r1] + F * (population[r2] - population[i])
                
                # Crossover (Binomial)
                trial = population[i].copy()
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                    
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                ft = func(trial)
                calls_used += 1
                update_best(trial, ft)
                
                # Selection (Greedy)
                if ft <= fitness[i]:
                    population[i] = trial
                    fitness[i] = ft

        return x_best, f_best
# EVOLVE-BLOCK-END
