"""Differential Evolution optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with population-based search and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # 15s safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def eval_func(x):
            nonlocal calls_made, best_f, best_x
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Population size λ=5 for DE/rand/1/bin
        pop_size = 5
        crossover_rate = 0.7
        F = 0.8  # Differential weight

        # Initialize population with diverse starting points
        # Use structured starts first, then random
        starts = [
            np.zeros(dim),
            np.full(dim, 2.0),
            np.full(dim, -2.0),
            np.array([3.0] * dim),
            np.array([-3.0] * dim),
        ]
        
        # Add random starts to fill population if needed
        while len(starts) < pop_size:
            starts.append(rng.uniform(-4, 4, dim))

        # Shuffle starts for diversity
        idx = rng.permutation(pop_size)
        starts = [starts[i] for i in idx]

        # Evaluate initial population
        population = []
        for start in starts[:pop_size]:
            if calls_made >= budget or time.time() > deadline:
                break
            fx = eval_func(start)
            population.append((start.copy(), fx))

        if len(population) < pop_size:
            # If we couldn't fill the population, use what we have
            pass

        # Differential Evolution iterations
        # Each iteration: for each individual, create a trial vector
        # DE/rand/1/bin: v = x_r1 + F * (x_r2 - x_r3)
        
        generation = 0
        max_generations = 100  # Safety limit, but we're budget-constrained
        
        while calls_made < budget - 1 and time.time() < deadline:
            generation += 1
            
            # Sort population by fitness (best first)
            population.sort(key=lambda x: x[1])
            
            # For each individual in the population, create a trial vector
            new_population = []
            
            for i in range(len(population)):
                if calls_made >= budget - 1 or time.time() > deadline:
                    break
                    
                x_i, f_i = population[i]
                
                # Select three distinct random indices different from i
                # We need at least 4 distinct individuals for DE/rand/1
                if len(population) < 4:
                    # Fallback: use simple mutation if population too small
                    r1 = rng.integers(0, len(population))
                    r2 = rng.integers(0, len(population))
                    r3 = rng.integers(0, len(population))
                    if r1 == i:
                        r1 = (r1 + 1) % len(population)
                    if r2 == i or r2 == r1:
                        r2 = (r2 + 1) % len(population)
                    if r3 == i or r3 == r1 or r3 == r2:
                        r3 = (r3 + 1) % len(population)
                else:
                    # Select 3 distinct random indices, none equal to i
                    candidates = [j for j in range(len(population)) if j != i]
                    r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
                
                x_r1 = population[r1][0]
                x_r2 = population[r2][0]
                x_r3 = population[r3][0]
                
                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                v = x_r1 + F * (x_r2 - x_r3)
                
                # Binomial crossover
                # Random starting point and crossover rate
                j_rand = rng.integers(0, dim)
                u = rng.uniform(0, 1, dim)
                
                trial = x_i.copy()
                for j in range(dim):
                    if u[j] <= crossover_rate or j == j_rand:
                        trial[j] = v[j]
                
                # Clip to bounds
                trial = np.clip(trial, -5.0, 5.0)
                
                # Evaluate trial vector
                f_trial = eval_func(trial)
                
                # Selection: keep the better of trial and current
                if f_trial < f_i:
                    new_population.append((trial, f_trial))
                else:
                    new_population.append((x_i, f_i))
            
            # Update population with new individuals
            # Keep only the best pop_size individuals
            new_population.sort(key=lambda x: x[1])
            population = new_population[:pop_size]
            
            # Check for convergence
            if len(population) > 0:
                f_best_current = population[0][1]
                f_worst_current = population[-1][1]
                if abs(f_worst_current - f_best_current) < 1e-10:
                    break

        # Final refinement: if we have budget left, do a local search around the best point
        if best_x is not None and calls_made < budget - 5 and time.time() < deadline:
            # Simple local search with decreasing step size
            x = best_x.copy()
            fx = best_f
            step_sizes = [0.5, 0.25, 0.1, 0.05, 0.01]
            
            for step in step_sizes:
                if calls_made >= budget - 1 or time.time() > deadline:
                    break
                
                improved = True
                while improved and calls_made < budget - 1 and time.time() < deadline:
                    improved = False
                    for d in range(dim):
                        for direction in [1, -1]:
                            trial = x.copy()
                            trial[d] += direction * step
                            trial = np.clip(trial, -5.0, 5.0)
                            
                            if np.allclose(trial, x, atol=1e-10):
                                continue
                            
                            f_trial = eval_func(trial)
                            if f_trial < fx - 1e-12:
                                x = trial
                                fx = f_trial
                                improved = True
                                break
                        if improved:
                            break

        # Ensure we have a valid result
        if best_x is None:
            x = np.zeros(dim)
            best_x = x
            best_f = func(x)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
