import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 300s limit
        budget = self.budget
        dim = self.dim
        
        calls = [0]
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        def evaluate(x):
            nonlocal best_x, best_f
            if calls[0] >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            calls[0] += 1
            x = np.clip(x, -5, 5)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)
        
        # Structural change: Population-based Evolution Strategy (μ+λ)-ES
        # This replaces the sequential (1+1)-ES restarts with a population-based search.
        # The population allows for better exploration of the multimodal landscape of Lunacek
        # by maintaining diversity and using selection pressure to find good basins.
        # We also incorporate a "restart" mechanism where if the population stagnates,
        # we inject new random individuals to escape local optima.

        # Parameters
        pop_size = 20  # μ
        offspring_size = 40  # λ
        sigma = 1.0  # Initial step size
        sigma_min = 1e-6
        sigma_max = 10.0
        momentum_decay = 0.9
        
        # State
        population = []
        fitnesses = []
        
        # Phase 1: Initialize population with random points
        # Use a mix of random points and points near the origin/best known
        for i in range(pop_size):
            if i == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-5, 5, dim)
            f = evaluate(x)
            population.append(x)
            fitnesses.append(f)
            
        # Sort initial population
        idx = np.argsort(fitnesses)
        population = [population[i] for i in idx]
        fitnesses = [fitnesses[i] for i in idx]
        
        # Update best
        if fitnesses[0] < best_f:
            best_f = fitnesses[0]
            best_x = population[0].copy()
            
        momentum = np.zeros(dim)
        stagnation_counter = 0
        best_f_so_far = best_f
        
        # Phase 2: Evolutionary loop
        while calls[0] < budget and time.time() < deadline:
            # Generate offspring
            offspring = []
            offspring_fitnesses = []
            
            for _ in range(offspring_size):
                if calls[0] >= budget or time.time() > deadline:
                    break
                
                # Select parent from population (tournament selection of size 2)
                i1 = rng.integers(0, pop_size)
                i2 = rng.integers(0, pop_size)
                if fitnesses[i1] < fitnesses[i2]:
                    parent = population[i1]
                else:
                    parent = population[i2]
                
                # Create child
                z = rng.standard_normal(dim)
                # Apply momentum to direction
                momentum = momentum_decay * momentum + (1 - momentum_decay) * z
                child = parent + sigma * momentum
                
                # Clip to bounds
                child = np.clip(child, -5, 5)
                
                f_child = evaluate(child)
                offspring.append(child)
                offspring_fitnesses.append(f_child)
            
            if not offspring:
                break
                
            # Combine parents and offspring
            combined_pop = population + offspring
            combined_fit = fitnesses + offspring_fitnesses
            
            # Select best pop_size individuals
            idx = np.argsort(combined_fit)
            new_population = [combined_pop[i] for i in idx[:pop_size]]
            new_fitnesses = [combined_fit[i] for i in idx[:pop_size]]
            
            # Update best
            if new_fitnesses[0] < best_f:
                best_f = new_fitnesses[0]
                best_x = new_population[0].copy()
                
            # Check for stagnation
            if new_fitnesses[0] < best_f_so_far - 1e-6:
                best_f_so_far = new_fitnesses[0]
                stagnation_counter = 0
                # Slightly increase sigma if we are making progress
                sigma = min(sigma * 1.05, sigma_max)
            else:
                stagnation_counter += 1
                # Decrease sigma if stagnant
                sigma = max(sigma * 0.95, sigma_min)
                
                # If severely stagnant, restart some individuals
                if stagnation_counter > 20:
                    # Replace worst half of population with new random points
                    n_restart = pop_size // 2
                    for i in range(pop_size - n_restart, pop_size):
                        new_x = rng.uniform(-5, 5, dim)
                        new_f = evaluate(new_x)
                        new_population[i] = new_x
                        new_fitnesses[i] = new_f
                    
                    # Re-sort
                    idx = np.argsort(new_fitnesses)
                    new_population = [new_population[i] for i in idx]
                    new_fitnesses = [new_fitnesses[i] for i in idx]
                    
                    # Update best again
                    if new_fitnesses[0] < best_f:
                        best_f = new_fitnesses[0]
                        best_x = new_population[0].copy()
                        
                    stagnation_counter = 0
                    # Reset momentum
                    momentum = np.zeros(dim)
                    # Reset sigma to a medium value
                    sigma = 1.0

            population = new_population
            fitnesses = new_fitnesses
            
            # Early termination if very close to 0 (Lunacek min is often 0 or close)
            # But we don't know f_opt, so we just check if improvement is tiny
            if best_f < 1e-4:
                # Do a few more local steps with very small sigma
                sigma = 0.01
                if stagnation_counter > 10:
                    break

        # Final local refinement around best_x
        if calls[0] < budget and time.time() < deadline:
            sigma = 0.1
            x = best_x.copy()
            f = best_f
            for _ in range(50):
                if calls[0] >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                x_new = x + sigma * z
                x_new = np.clip(x_new, -5, 5)
                f_new = evaluate(x_new)
                if f_new < f:
                    x = x_new
                    f = f_new
                    sigma *= 0.9
                else:
                    sigma *= 0.8
                if sigma < 1e-5:
                    break

        if calls[0] == 0:
            evaluate(np.zeros(dim))

        return best_x, best_f
