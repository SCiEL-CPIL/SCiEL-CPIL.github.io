"""Optimizer for BBOB-style f4 (Büche-Rastrigin) function."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution optimizer with strict budget tracking.
    
    Structural change: Replaces SciPy Nelder-Mead with a custom Differential Evolution
    implementation. This ensures strict adherence to the function call budget (500 calls)
    and leverages population-based search which is effective for multimodal functions
    like Büche-Rastrigin.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        global_best_x = None
        global_best_fx = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, global_best_x, global_best_fx
            if calls_made >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < global_best_fx:
                global_best_fx = fx
                global_best_x = x.copy()
            return fx

        # Population size for DE
        # In 5D, pop_size of 10-20 is reasonable.
        # Let's use pop_size = 15.
        # Total calls = pop_size (init) + generations * pop_size (mutations)
        # Budget = 500.
        # If pop_size = 15, we can afford ~30 generations.
        pop_size = 15
        population = rng.uniform(-5, 5, size=(pop_size, dim))
        fitnesses = np.full(pop_size, np.inf)

        # Evaluate initial population
        for i in range(pop_size):
            fitnesses[i] = evaluate(population[i])
            if calls_made >= budget:
                break

        # Sort population initially? DE usually maintains a list.
        # Let's keep it simple: maintain population and fitness arrays.
        # We need to track the best individual in the population.

        # DE parameters
        F = 0.8  # Mutation scale
        CR = 0.5  # Crossover rate

        while calls_made < budget and time.time() < deadline:
            # Check if we have enough budget for a full generation
            if budget - calls_made < pop_size:
                break
            
            # Sort population by fitness to find best
            idx_sorted = np.argsort(fitnesses)
            best_idx = idx_sorted[0]
            best_x = population[best_idx].copy()
            best_f = fitnesses[best_idx]

            # Generate offspring for each individual
            for i in range(pop_size):
                if calls_made >= budget or time.time() > deadline:
                    break
                
                # Pick 3 distinct random indices different from i
                # Simple approach: pick 3 random indices from 0..pop_size-1
                # Ensure they are distinct and not i
                candidates = np.arange(pop_size)
                candidates = candidates[candidates != i]
                if len(candidates) < 3:
                    break
                a_idx, b_idx, c_idx = rng.choice(candidates, size=3, replace=False)
                
                # Mutation: v = x_a + F * (x_b - x_c)
                v = population[a_idx] + F * (population[b_idx] - population[c_idx])
                
                # Crossover
                trial = population[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate trial
                f_trial = evaluate(trial)
                
                # Selection
                if f_trial <= fitnesses[i]:
                    population[i] = trial
                    fitnesses[i] = f_trial

        return global_best_x, global_best_fx
# EVOLVE-BLOCK-END
