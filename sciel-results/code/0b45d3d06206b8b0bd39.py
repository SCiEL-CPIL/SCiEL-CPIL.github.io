"""Optimizer for BBOB-style Rosenbrock (f8) on [-5,5]^5."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid Differential Evolution and Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 285  # Safety margin

        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def safe_call(x):
            nonlocal calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            if calls >= budget:
                return np.inf
            f = func(x)
            if not np.isfinite(f):
                f = np.inf
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Global Exploration with Differential Evolution
        # Small population to allow more generations within budget
        pop_size = 20
        max_generations = min(20, (budget - 50) // pop_size)  # Reserve 50 for local polish
        if max_generations < 2:
            # If budget is too small for DE, just do random search + local
            max_generations = 1
            pop_size = 10

        # Initialize population uniformly in [-5, 5]^dim
        population = rng.uniform(-5.0, 5.0, size=(pop_size, dim))
        fitness = np.array([safe_call(x) for x in population])
        
        # Sort initial population
        idx = np.argsort(fitness)
        population = population[idx]
        fitness = fitness[idx]
        
        # DE Parameters
        F = 0.8  # Scale factor
        CR = 0.5  # Crossover rate
        
        for gen in range(max_generations):
            if calls >= budget or time.time() > deadline:
                break
            
            new_population = []
            new_fitness = []
            
            for i in range(pop_size):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Mutation: DE/rand/1
                r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                # Ensure r1, r2, r3 are distinct from i if possible, but standard DE allows any
                v = population[r1] + F * (population[r2] - population[r3])
                
                # Crossover: Binomial
                u = rng.random(dim)
                j_rand = rng.integers(dim)
                trial = np.where(u < CR, v, population[i])
                trial[j_rand] = v[j_rand] # Ensure at least one gene from mutant
                
                # Selection
                f_trial = safe_call(trial)
                
                if f_trial <= fitness[i]:
                    new_population.append(trial)
                    new_fitness.append(f_trial)
                else:
                    new_population.append(population[i])
                    new_fitness.append(fitness[i])
            
            if not new_population:
                break
                
            population = np.array(new_population)
            fitness = np.array(new_fitness)
            
            # Update best
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_f:
                best_f = fitness[current_best_idx]
                best_x = population[current_best_idx].copy()

        # Phase 2: Local Refinement with Nelder-Mead
        # Use remaining budget for local search from the best point found
        remaining_budget = budget - calls
        if remaining_budget > 10 and time.time() < deadline:
            try:
                # Nelder-Mead is effective for final convergence in the valley
                result = minimize(
                    safe_call,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'maxfev': remaining_budget,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        return best_x, best_f
