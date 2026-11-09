"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution optimizer for BBOB-style f17 (Schaffers F7)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 300s limit, leave 15s margin
        
        rng = np.random.default_rng(0)
        best_x = None
        best_fx = np.inf
        calls_made = 0
        
        def call_func(x):
            nonlocal calls_made
            fx = func(x)
            calls_made += 1
            return fx
        
        n = self.dim
        
        # DE parameters
        pop_size = 10  # population size
        F = 0.8  # differential weight
        CR = 0.9  # crossover rate
        
        # Phase 1: Initialize population
        population = np.zeros((pop_size, n))
        fitness = np.zeros(pop_size)
        
        for i in range(pop_size):
            if time.time() > deadline or calls_made >= self.budget:
                break
            x = rng.uniform(-5, 5, n)
            fx = call_func(x)
            population[i] = x
            fitness[i] = fx
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
        
        if best_x is None:
            best_x = np.zeros(n)
            best_fx = call_func(best_x)
        
        # Phase 2: Differential Evolution
        stagnation_count = 0
        max_stagnation = 15  # restart after this many generations without improvement
        
        while calls_made < self.budget and time.time() < deadline:
            if calls_made + pop_size > self.budget:
                # Use remaining budget for local refinement
                break
            
            prev_best = best_fx
            new_population = population.copy()
            new_fitness = fitness.copy()
            
            for i in range(pop_size):
                if calls_made >= self.budget or time.time() > deadline:
                    break
                
                # Select three distinct random indices different from i
                candidates = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(candidates, size=3, replace=False)
                
                # Create trial vector using DE/rand/1
                v = population[a] + F * (population[b] - population[c])
                v = np.clip(v, -5, 5)
                
                # Crossover
                r = rng.random()
                j_rand = rng.integers(n)
                trial = np.where(rng.random(n) <= CR, v, population[i])
                trial[j_rand] = v[j_rand]  # Ensure at least one gene from v
                trial = np.clip(trial, -5, 5)
                
                # Evaluate trial
                trial_fx = call_func(trial)
                
                # Selection
                if trial_fx <= fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = trial_fx
                    if trial_fx < best_fx:
                        best_fx = trial_fx
                        best_x = trial.copy()
            
            population = new_population
            fitness = new_fitness
            
            # Check for stagnation
            if best_fx >= prev_best - 1e-10:
                stagnation_count += 1
            else:
                stagnation_count = 0
            
            # Random restart if stagnated
            if stagnation_count >= max_stagnation:
                stagnation_count = 0
                # Perturb some population members
                for _ in range(pop_size // 2):
                    idx = rng.integers(pop_size)
                    population[idx] = rng.uniform(-5, 5, n)
                    fitness[idx] = call_func(population[idx])
                    if fitness[idx] < best_fx:
                        best_fx = fitness[idx]
                        best_x = population[idx].copy()
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
