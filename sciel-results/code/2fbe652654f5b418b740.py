"""Differential Evolution with adaptive parameters and local polishing for BBOB f16."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s kill

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        n_calls = 0

        def check_time():
            return time.time() > deadline

        def call_func(x):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget:
                return None, None
            n_calls += 1
            try:
                x_clipped = np.clip(np.asarray(x, dtype=float), -5, 5)
                f_val = float(func(x_clipped))
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clipped.copy()
                return f_val, x_clipped
            except Exception:
                return None, None

        rng = np.random.default_rng(42)

        # Phase 1: Initialize Population
        pop_size = 20
        population = []
        pop_fitness = []

        # Use a mix of random and grid-like initialization for better coverage
        for i in range(pop_size):
            if check_time() or n_calls >= budget:
                break
            # Stratified random initialization
            x = rng.uniform(-5, 5, dim)
            f_val, x_eval = call_func(x)
            if f_val is not None:
                population.append(x_eval)
                pop_fitness.append(f_val)

        if not population:
            x = np.zeros(dim)
            call_func(x)
            if best_x is None:
                return None
            population.append(best_x)
            pop_fitness.append(best_f)

        population = np.array(population)
        pop_fitness = np.array(pop_fitness)

        # Phase 2: Differential Evolution
        F = 0.8  # Scale factor
        CR = 0.5  # Crossover rate
        stagnation_count = 0
        best_f_prev = best_f

        # Adaptive parameters
        min_F = 0.3
        max_F = 1.0
        min_CR = 0.1
        max_CR = 0.9

        while n_calls < budget and not check_time():
            # Adapt F and CR based on stagnation
            if stagnation_count > 10:
                # Increase exploration
                F = np.clip(F + 0.1, min_F, max_F)
                CR = np.clip(CR + 0.1, min_CR, max_CR)
                stagnation_count = 0
            elif stagnation_count > 5:
                # Moderate exploration
                F = np.clip(F + 0.05, min_F, max_F)
                CR = np.clip(CR + 0.05, min_CR, max_CR)
            
            # Perform DE iteration for each individual in population
            for i in range(len(population)):
                if check_time() or n_calls >= budget:
                    break
                
                # Select 3 distinct individuals other than i
                idxs = [j for j in range(len(population)) if j != i]
                r1, r2, r3 = rng.choice(idxs, 3, replace=False)
                
                # Mutation: DE/rand/1
                v = population[r1] + F * (population[r2] - population[r3])
                
                # Crossover: Binomial crossover
                j_rand = rng.integers(0, dim)
                trial = population[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_val, x_eval = call_func(trial)
                if f_val is None:
                    continue
                
                # Selection
                if f_val <= pop_fitness[i]:
                    population[i] = x_eval
                    pop_fitness[i] = f_val
                    
                    if f_val < best_f_prev:
                        best_f_prev = f_val
                        stagnation_count = 0
                    else:
                        stagnation_count += 1
                else:
                    stagnation_count += 1

        # Phase 3: Local Polishing with Nelder-Mead
        if n_calls < budget and not check_time():
            def objective(x):
                nonlocal n_calls
                if n_calls >= budget:
                    return np.inf
                x_clipped = np.clip(x, -5, 5)
                f_val, _ = call_func(x_clipped)
                return f_val if f_val is not None else np.inf

            # Try polishing from the best individual and a few others
            candidates = [best_x]
            # Add top 2 from population if distinct
            sorted_indices = np.argsort(pop_fitness)
            for idx in sorted_indices[:3]:
                if not np.allclose(population[idx], best_x):
                    candidates.append(population[idx])

            for x0 in candidates:
                if check_time() or n_calls >= budget:
                    break
                
                try:
                    # Simple coordinate descent or Nelder-Mead
                    # Using scipy if available, otherwise simple gradient-free local search
                    # Since we want to avoid heavy dependencies if possible, but scipy is allowed.
                    # Let's use a simple local search to be safe and fast.
                    current_x = x0.copy()
                    current_f = objective(current_x)
                    
                    step_size = 0.1
                    for _ in range(10): # Limit local search steps
                        if check_time() or n_calls >= budget:
                            break
                        improved = False
                        for d in range(dim):
                            for direction in [-1, 1]:
                                new_x = current_x.copy()
                                new_x[d] += direction * step_size
                                new_x = np.clip(new_x, -5, 5)
                                new_f = objective(new_x)
                                if new_f < current_f:
                                    current_x = new_x
                                    current_f = new_f
                                    improved = True
                            if improved:
                                break
                        if not improved:
                            step_size *= 0.5
                            if step_size < 1e-3:
                                break
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
