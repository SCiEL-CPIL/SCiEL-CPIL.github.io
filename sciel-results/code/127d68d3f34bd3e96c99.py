"""Hybrid optimizer: DE exploration + Local exploitation."""
import time
import numpy as np


class Optimizer:
    """Hybrid: DE for global search, Coordinate Descent for local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= budget or time.time() > deadline - 10.0:
                return None
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            update_best(x, f)
            return f

        # Phase 1: Differential Evolution
        pop_size = 10
        F = 0.8
        CR = 0.5
        de_budget = int(budget * 0.7)  # Use 70% of budget for DE
        
        # Initialize population
        population = [rng.uniform(-5, 5, dim) for _ in range(pop_size)]
        fitnesses = []
        for i, x in enumerate(population):
            f = safe_call(x)
            if f is None:
                return best_x, best_f
            fitnesses.append(f)
        
        # DE Loop
        while calls_made < de_budget and time.time() < deadline - 50.0:
            # Sort population by fitness
            indices = np.argsort(fitnesses)
            population = [population[i] for i in indices]
            fitnesses = [fitnesses[i] for i in indices]

            # Generate offspring
            new_pop = population.copy()
            new_fit = fitnesses.copy()
            
            for i in range(pop_size):
                if calls_made >= de_budget or time.time() > deadline - 50.0:
                    break
                
                # Select 3 distinct random individuals
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                
                # Mutation
                mutant = population[r1] + F * (population[r2] - population[r3])
                
                # Crossover
                trial = np.zeros(dim)
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                    else:
                        trial[j] = population[i][j]
                
                trial = np.clip(trial, -5, 5)
                f_trial = safe_call(trial)
                if f_trial is None:
                    break
                
                # Selection
                if f_trial <= fitnesses[i]:
                    new_pop[i] = trial
                    new_fit[i] = f_trial
            
            population = new_pop
            fitnesses = new_fit

        # Phase 2: Local Refinement using Coordinate Descent on top candidates
        remaining_budget = budget - calls_made
        if remaining_budget > 10 and time.time() < deadline - 20.0:
            # Get top 2 unique points from the final population
            indices = np.argsort(fitnesses)
            top_candidates = [population[i].copy() for i in indices[:2]]
            
            if best_x is not None:
                top_candidates.append(best_x.copy())
            
            # Deduplicate roughly
            unique_candidates = []
            for c in top_candidates:
                is_unique = True
                for uc in unique_candidates:
                    if np.linalg.norm(c - uc) < 1e-6:
                        is_unique = False
                        break
                if is_unique:
                    unique_candidates.append(c)
            
            # Allocate remaining budget to local search
            for c in unique_candidates[:2]:
                if calls_made >= budget or time.time() > deadline - 10.0:
                    break
                
                x = c.copy()
                fx = safe_call(x)
                if fx is None:
                    break
                
                # Adaptive step size Coordinate Descent
                step = 1.0
                improved = True
                max_iters = 50
                
                for _ in range(max_iters):
                    if calls_made >= budget or time.time() > deadline - 5.0:
                        break
                    if not improved:
                        step *= 0.5
                        if step < 1e-4:
                            break
                        improved = False
                        continue
                    
                    for i in range(dim):
                        if calls_made >= budget or time.time() > deadline - 5.0:
                            break
                        
                        # Try +step
                        x_new = x.copy()
                        x_new[i] = np.clip(x_new[i] + step, -5, 5)
                        f_new = safe_call(x_new)
                        if f_new is not None and f_new < fx:
                            x = x_new
                            fx = f_new
                            improved = True
                            continue
                            
                        # Try -step
                        x_new = x.copy()
                        x_new[i] = np.clip(x_new[i] - step, -5, 5)
                        f_new = safe_call(x_new)
                        if f_new is not None and f_new < fx:
                            x = x_new
                            fx = f_new
                            improved = True

        return best_x, best_f
