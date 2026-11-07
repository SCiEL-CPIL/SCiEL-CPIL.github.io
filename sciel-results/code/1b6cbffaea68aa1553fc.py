import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0
        
        try:
            # Differential Evolution parameters
            pop_size = 20
            F = 0.8  # Differential weight
            CR = 0.9  # Crossover rate
            
            # Initialize population
            population = rng.uniform(-5, 5, (pop_size, dim))
            fitness = np.array([func(x) for x in population])
            calls_made += pop_size
            
            best_idx = np.argmin(fitness)
            best_x = population[best_idx].copy()
            best_f = fitness[best_idx]
            
            # DE loop
            while calls_made < budget and time.time() < deadline:
                # Check time
                if time.time() > deadline - 0.5:
                    break
                    
                new_population = []
                new_fitness = []
                
                for i in range(pop_size):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    
                    # Select three distinct random indices
                    r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                    while r1 == i or r2 == i or r3 == i:
                        r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                    
                    # Create trial vector using DE/rand/1/bin
                    v = population[r1] + F * (population[r2] - population[r3])
                    
                    # Bounded crossover
                    j_rand = rng.integers(dim)
                    for j in range(dim):
                        if rng.random() < CR or j == j_rand:
                            trial_j = v[j]
                        else:
                            trial_j = population[i][j]
                        new_population.append(trial_j) if len(new_population) == 0 else None
                    
                    # Actually, let me fix the trial vector creation
                    trial = np.zeros(dim)
                    for j in range(dim):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = v[j]
                        else:
                            trial[j] = population[i][j]
                    
                    trial = np.clip(trial, -5, 5)
                    f_trial = func(trial)
                    calls_made += 1
                    
                    # Selection
                    if f_trial <= fitness[i]:
                        new_population.append(trial)
                        new_fitness.append(f_trial)
                    else:
                        new_population.append(population[i])
                        new_fitness.append(fitness[i])
                
                if len(new_population) < pop_size:
                    break
                
                population = np.array(new_population)
                fitness = np.array(new_fitness)
                
                # Update best
                best_idx = np.argmin(fitness)
                if fitness[best_idx] < best_f:
                    best_f = fitness[best_idx]
                    best_x = population[best_idx].copy()
                    
                # Adaptive parameter tuning based on progress
                progress = (best_f - np.min(fitness)) / (abs(best_f) + 1e-10)
                if progress < 1e-6 and calls_made > budget * 0.3:
                    # Reduce F and increase CR for exploitation
                    F = max(0.3, F * 0.9)
                    CR = min(0.99, CR + 0.01)
                    
        except Exception:
            pass

        return best_x, best_f
