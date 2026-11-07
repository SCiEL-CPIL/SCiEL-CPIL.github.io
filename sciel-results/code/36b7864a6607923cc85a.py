"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Hybrid Differential Evolution (DE) with Integer-Mixed Initialization
    and Final Local Polish for Rastrigin rotated.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        calls = 0
        rng = np.random.default_rng(42)

        # Population size for DE
        pop_size = 20
        # F and CR parameters for DE
        F = 0.8
        CR = 0.9

        # Phase 1: Initialization
        # Mix of random uniform and integer points
        # Integer points are good candidates for Rastrigin minima
        init_points = []
        
        # 50% Random Uniform
        for _ in range(pop_size // 2):
            x = rng.uniform(-5, 5, size=dim)
            init_points.append(x)
        
        # 50% Random Integer (clipped to [-4, 4] to stay in basin)
        for _ in range(pop_size - pop_size // 2):
            x = rng.integers(-4, 5, size=dim).astype(float)
            init_points.append(x)

        population = np.array(init_points)
        
        # Evaluate initial population
        fitness = np.array([func(x) for x in population])
        calls += pop_size
        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_f = fitness[best_idx]

        # Phase 2: Differential Evolution
        # Target: use ~450 calls total for DE
        max_de_calls = 450
        current_calls = calls
        
        while current_calls + pop_size <= max_de_calls and time.time() < deadline:
            new_population = []
            new_fitness = []
            
            for i in range(pop_size):
                # Select distinct random indices
                while True:
                    r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                    if r1 != i and r2 != i and r3 != i:
                        break
                
                # DE/rand/1/bin
                v = population[r1] + F * (population[r2] - population[r3])
                
                # Binary crossover
                j_rand = rng.integers(0, dim)
                trial = population[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Clip to box
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                calls += 1
                current_calls += 1
                
                # Selection
                if f_trial <= fitness[i]:
                    new_population.append(trial)
                    new_fitness.append(f_trial)
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                else:
                    new_population.append(population[i])
                    new_fitness.append(fitness[i])
            
            population = np.array(new_population)
            fitness = np.array(new_fitness)
            
            if time.time() >= deadline:
                break

        # Phase 3: Local Polish
        # Use remaining budget for a fast local search from best_x
        remaining_budget = budget - calls
        if remaining_budget > 10 and time.time() < deadline:
            # Simple (1+1)-ES or Nelder-Mead
            # Let's use a simple adaptive (1+1)-ES
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            no_improve = 0
            
            for _ in range(remaining_budget):
                if time.time() >= deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.1
                    no_improve = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
                    no_improve += 1
                    if no_improve > 20:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
