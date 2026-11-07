"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based evolutionary strategy with adaptive step size for BBOB f13 Sharp Ridge."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # safety margin before 300s limit

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_used = 0

        def safe_call(x):
            nonlocal calls_used, best_f, best_x
            if calls_used >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            calls_used += 1
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        rng = np.random.default_rng(12345)
        
        # Phase 1: Broad exploration
        f0 = safe_call(np.zeros(dim))
        
        n_explore = 40
        for i in range(n_explore - 1):
            if calls_used >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4.0, 4.0, dim)
            safe_call(x)
        
        # Phase 2: Population-based refinement with proper fitness tracking
        pop_size = 20
        mu = 10
        
        # Initialize population around best point
        sigma = 1.5
        population = []
        pop_fitness = []
        
        # Add best point
        population.append(best_x.copy())
        pop_fitness.append(best_f)
        
        # Add mutated points
        for i in range(pop_size - 1):
            if calls_used >= budget or time.time() > deadline:
                break
            x = best_x + sigma * rng.standard_normal(dim)
            f = safe_call(x)
            if f is not None:
                population.append(x)
                pop_fitness.append(f)
        
        no_improve_count = 0
        max_no_improve = 15
        
        while calls_used < budget and time.time() < deadline:
            # Generate offspring
            offspring = []
            off_fitness = []
            
            for i in range(pop_size):
                if calls_used >= budget or time.time() > deadline:
                    break
                # Select parent (tournament from top mu)
                top_indices = np.argsort(pop_fitness)[:min(mu, len(population))]
                parent_idx = top_indices[rng.integers(0, len(top_indices))]
                parent = population[parent_idx]
                child = parent + sigma * rng.standard_normal(dim)
                f = safe_call(child)
                if f is not None:
                    offspring.append(child)
                    off_fitness.append(f)
            
            if len(offspring) < 5:
                break
            
            # Combine and select
            combined = population + offspring
            combined_f = pop_fitness + off_fitness
            
            # Select best pop_size
            indices = np.argsort(combined_f)
            new_population = [combined[i] for i in indices[:pop_size]]
            new_pop_fitness = [combined_f[i] for i in indices[:pop_size]]
            
            # Check improvement
            current_best = min(combined_f)
            if current_best < best_f - 1e-10:
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Adaptive sigma
            if no_improve_count >= max_no_improve:
                sigma *= 0.7
                no_improve_count = 0
            elif no_improve_count == 0:
                sigma = min(sigma * 1.05, 5.0)
            
            sigma = np.clip(sigma, 0.01, 5.0)
            population = new_population
            pop_fitness = new_pop_fitness
        
        # Phase 3: Final local refinement
        if calls_used < budget and time.time() < deadline:
            local_sigma = 0.3
            for _ in range(30):
                if calls_used >= budget or time.time() > deadline:
                    break
                x = best_x + local_sigma * rng.standard_normal(dim)
                f = safe_call(x)
                if f is not None and f < best_f:
                    local_sigma = max(local_sigma * 0.9, 0.01)
                else:
                    local_sigma *= 0.95
                local_sigma = max(local_sigma, 0.01)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
