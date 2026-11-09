"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with adaptive parameters for f22."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 320.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        def check_time():
            return (time.time() - start_time) < time_limit

        # Phase 1: Random sampling to find a good starting region
        n_random = min(50, budget)
        x_best = rng.uniform(-5, 5, dim)
        f_best = func(x_best)
        calls = 1

        for _ in range(n_random - 1):
            if not check_time():
                break
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            calls += 1
            if f < f_best:
                f_best, x_best = f, x

        # Phase 2: Differential Evolution
        remaining = budget - calls
        if remaining <= 0:
            return x_best, f_best

        # Population size: typically 5-10 times dim for DE
        pop_size = max(10, 5 * dim)
        
        # Initialize population: mix of random points and perturbations of best
        population = np.empty((pop_size, dim))
        fitness = np.empty(pop_size)
        
        # First individual is the best found
        population[0] = x_best
        fitness[0] = f_best
        
        # Rest are random or perturbed
        for i in range(1, pop_size):
            if not check_time() or calls >= budget:
                break
            if i < pop_size // 2:
                # Perturb the best point
                population[i] = np.clip(x_best + 1.5 * rng.standard_normal(dim), -5, 5)
            else:
                # Random points
                population[i] = rng.uniform(-5, 5, dim)
            
            fitness[i] = func(population[i])
            calls += 1
            if fitness[i] < f_best:
                f_best, x_best = fitness[i], population[i].copy()

        # DE parameters (adaptive)
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate
        
        # Main DE loop
        while calls < budget and check_time():
            # Sort population by fitness
            idx = np.argsort(fitness)
            population = population[idx]
            fitness = fitness[idx]
            
            # Update best
            if fitness[0] < f_best:
                f_best, x_best = fitness[0], population[0].copy()
            
            # Generate new population
            new_population = population.copy()
            new_fitness = fitness.copy()
            
            for i in range(1, pop_size):
                if not check_time() or calls >= budget:
                    break
                    
                # Select three distinct random individuals different from i
                r1, r2, r3 = rng.choice([j for j in range(pop_size) if j != i], 3, replace=False)
                
                # Differential variation
                trial = population[r1] + F * (population[r2] - population[r3])
                
                # Binary crossover
                cross_point = rng.integers(0, dim)
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = population[i][j]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate trial
                f_trial = func(trial)
                calls += 1
                
                # Selection
                if f_trial <= fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = f_trial
                    if f_trial < f_best:
                        f_best, x_best = f_trial, trial.copy()
            
            population = new_population
            fitness = new_fitness
            
            # Adaptive parameter adjustment
            # Reduce F and CR slightly as we progress to exploit
            progress = min(1.0, calls / budget)
            F = 0.8 * (1 - 0.3 * progress)
            CR = 0.9 * (1 - 0.2 * progress)
            
            # Restart if stagnated (no improvement in last 20 evaluations)
            if calls % 20 == 0:
                # Check if we've made progress recently
                recent_improvement = False
                for i in range(pop_size):
                    if fitness[i] < f_best + 1e-6:
                        recent_improvement = True
                        break
                
                if not recent_improvement and rng.random() < 0.3:
                    # Restart with new random population around best
                    for i in range(1, pop_size):
                        if not check_time() or calls >= budget:
                            break
                        population[i] = np.clip(x_best + 2.0 * rng.standard_normal(dim), -5, 5)
                        fitness[i] = func(population[i])
                        calls += 1
                        if fitness[i] < f_best:
                            f_best, x_best = fitness[i], population[i].copy()

        return x_best, f_best
# EVOLVE-BLOCK-END
