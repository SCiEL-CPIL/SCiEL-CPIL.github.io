import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Reserve budget for local search
        local_search_budget = min(50, budget // 10)
        de_budget = budget - local_search_budget
        
        pop_size = 30  # Increased from 20 for better diversity
        
        # Initialize with Latin hypercube-like sampling for better coverage
        population = np.empty((pop_size, dim))
        for i in range(pop_size):
            for j in range(dim):
                population[i, j] = -5 + 10 * ((i + 0.5) / pop_size + rng.random() / pop_size)
        
        fitnesses = np.array([func(x) for x in population])
        calls_made = pop_size
        
        best_idx = np.argmin(fitnesses)
        best_x = population[best_idx].copy()
        best_f = fitnesses[best_idx]
        
        # Adaptive parameters
        F = 0.7
        CR = 0.8
        
        # Track stagnation
        no_improve_count = 0
        max_no_improve = 100
        
        while calls_made < de_budget:
            # Generate trials
            for i in range(pop_size):
                if calls_made >= de_budget:
                    break
                
                # Select r1, r2, r3 distinct from i and each other
                while True:
                    r1, r2, r3 = rng.integers(0, pop_size, 3)
                    if len(set([r1, r2, r3, i])) == 4:
                        break
                
                # Mutation: DE/rand/1
                v = population[r1] + F * (population[r2] - population[r3])
                
                # Crossover: binomial
                j_rand = rng.integers(0, dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = population[i][j]
                
                trial = np.clip(trial, -5, 5)
                f_trial = func(trial)
                calls_made += 1
                
                # Selection
                if f_trial < fitnesses[i]:
                    population[i] = trial
                    fitnesses[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            
            # Adaptive parameter adjustment
            if no_improve_count > max_no_improve:
                # Restart some of the worst individuals with random perturbations
                worst_indices = np.argsort(fitnesses)[-pop_size // 4:]
                for idx in worst_indices:
                    if calls_made >= de_budget:
                        break
                    # Perturb the worst individuals
                    population[idx] = best_x + rng.normal(0, 1.0, dim)
                    population[idx] = np.clip(population[idx], -5, 5)
                    fitnesses[idx] = func(population[idx])
                    calls_made += 1
                    if fitnesses[idx] < best_f:
                        best_f = fitnesses[idx]
                        best_x = population[idx].copy()
                no_improve_count = 0
                # Adjust parameters
                F = 0.5 + 0.3 * rng.random()
                CR = 0.6 + 0.3 * rng.random()
        
        # Local Search Phase
        remaining_calls = budget - calls_made
        
        if remaining_calls > 0:
            # Use Nelder-Mead with careful budget management
            res = minimize(
                func, 
                best_x, 
                method='Nelder-Mead',
                options={'maxfev': remaining_calls, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
        
        return best_x, best_f
