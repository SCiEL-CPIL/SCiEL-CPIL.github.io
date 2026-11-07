"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Population-based Differential Evolution (DE/rand/1/bin) for Büche-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        pop_size = 20
        dim = self.dim
        budget = self.budget

        # Initialize population uniformly in [-5, 5]^dim
        population = rng.uniform(-5, 5, (pop_size, dim))
        fitnesses = np.array([func(x) for x in population])
        
        best_idx = np.argmin(fitnesses)
        best_x = population[best_idx].copy()
        best_f = fitnesses[best_idx]
        
        # Track best value seen so far for early progress
        global_best_f = best_f
        global_best_x = best_x.copy()
        
        calls_made = pop_size
        remaining = budget - calls_made
        
        # Differential Evolution parameters
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate
        
        # Adaptive strategy: adjust F and CR based on progress
        # Büche-Rastrigin is multimodal; we need good exploration early and exploitation later
        
        while calls_made < budget:
            # Generate trial vectors for each member of the population
            new_population = np.zeros_like(population)
            new_fitnesses = np.zeros(pop_size)
            
            # Determine how many full generations we can do
            # Each generation uses pop_size evaluations
            # But we need to be careful not to exceed budget
            
            for i in range(pop_size):
                if calls_made >= budget:
                    break
                    
                # Select three distinct random indices different from i
                # DE/rand/1/bin
                while True:
                    a = rng.integers(0, pop_size)
                    b = rng.integers(0, pop_size)
                    c = rng.integers(0, pop_size)
                    if a != i and b != i and c != i and a != b and a != c:
                        break
                
                # Mutation: v = population[a] + F * (population[b] - population[c])
                v = population[a] + F * (population[b] - population[c])
                
                # Crossover: binary crossover
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial_j = v[j]
                    else:
                        trial_j = population[i][j]
                    new_population[i][j] = trial_j
                
                # Clip to box
                new_population[i] = np.clip(new_population[i], -5, 5)
                
                # Evaluate
                new_fitnesses[i] = func(new_population[i])
                calls_made += 1
                
                # Selection: greedy
                if new_fitnesses[i] < fitnesses[i]:
                    population[i] = new_population[i]
                    fitnesses[i] = new_fitnesses[i]
                    
                    if new_fitnesses[i] < global_best_f:
                        global_best_f = new_fitnesses[i]
                        global_best_x = new_population[i].copy()
            
            # Adaptive parameter control
            # Reduce exploration as we approach the end
            progress = calls_made / budget
            # Keep F relatively high for multimodal landscape, but reduce CR slightly
            # Actually, for Büche-Rastrigin, we might want to keep diversity high
            # Let's adapt F based on success rate
            # Simple adaptation: if progress > 0.5, reduce F slightly to exploit more
            if progress > 0.5:
                F = max(0.5, 0.8 - 0.3 * (progress - 0.5))
            else:
                F = 0.8
            
            # If we have budget left but not enough for a full generation,
            # do local searches or single-step improvements
            if calls_made < budget:
                # Use remaining budget for local refinement around the best solution
                # Simple random walk around the best
                while calls_made < budget:
                    # Generate a small perturbation
                    sigma = 0.5 * (1 - progress) + 0.1
                    candidate = global_best_x + sigma * rng.standard_normal(dim)
                    candidate = np.clip(candidate, -5, 5)
                    f_candidate = func(candidate)
                    calls_made += 1
                    
                    if f_candidate < global_best_f:
                        global_best_f = f_candidate
                        global_best_x = candidate.copy()
                    
                    # Also try to update the population member that is closest to best
                    # to maintain diversity near the best
                    distances = np.linalg.norm(population - global_best_x, axis=1)
                    closest_idx = np.argmin(distances)
                    if f_candidate < fitnesses[closest_idx]:
                        population[closest_idx] = candidate
                        fitnesses[closest_idx] = f_candidate
        
        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
