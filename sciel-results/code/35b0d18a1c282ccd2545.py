"""Rewritten optimizer using Differential Evolution with restarts."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution optimizer for BBOB f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _differential_evolution(self, func, budget, rng):
        dim = self.dim
        N = 25 
        F = 0.8  
        CR = 0.9  
        
        # Initialize population
        pop = rng.uniform(-5, 5, (N, dim))
        fitnesses = np.empty(N)
        calls = 0
        
        for i in range(N):
            fitnesses[i] = func(pop[i])
            calls += 1
            
        best_idx = np.argmin(fitnesses)
        best_x = pop[best_idx].copy()
        best_f = fitnesses[best_idx]
        
        no_improve_gens = 0
        max_no_improve_gens = 5 # Restart every 5 generations without improvement
        
        while calls + N <= budget:
            # We have enough budget for a full generation
            new_pop = pop.copy()
            new_fitnesses = fitnesses.copy()
            
            for i in range(N):
                # Choose distinct random indices
                r1, r2, r3 = rng.choice(N, 3, replace=False)
                
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                j_rand = rng.integers(0, dim)
                u = np.zeros(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = pop[i][j]
                        
                # Clip to bounds
                u = np.clip(u, -5, 5)
                
                # Evaluate
                f_u = func(u)
                calls += 1
                
                # Selection
                if f_u <= fitnesses[i]:
                    new_pop[i] = u
                    new_fitnesses[i] = f_u
            
            # Update population
            pop = new_pop
            fitnesses = new_fitnesses
            
            # Check improvement
            current_best_f = np.min(fitnesses)
            if current_best_f < best_f - 1e-10:
                best_f = current_best_f
                best_idx = np.argmin(fitnesses)
                best_x = pop[best_idx].copy()
                no_improve_gens = 0
            else:
                no_improve_gens += 1
                
            # Check for restart
            if no_improve_gens >= max_no_improve_gens:
                # Restart: Keep best, randomize others
                # We need to evaluate the new random points
                # This consumes N-1 calls
                if calls + (N - 1) > budget:
                    break # Not enough budget for restart evaluation
                
                pop[0] = best_x
                fitnesses[0] = best_f
                # Randomize the rest
                new_rand_pop = rng.uniform(-5, 5, (N-1, dim))
                new_rand_fit = np.empty(N-1)
                for i in range(N-1):
                    new_rand_fit[i] = func(new_rand_pop[i])
                    calls += 1
                    
                pop[1:] = new_rand_pop
                fitnesses[1:] = new_rand_fit
                
                # Update best if new randoms are better
                min_new_idx = 1 + np.argmin(fitnesses[1:])
                if fitnesses[min_new_idx] < best_f:
                    best_f = fitnesses[min_new_idx]
                    best_x = pop[min_new_idx].copy()
                    no_improve_gens = 0
                else:
                    no_improve_gens = 0 # Reset counter after restart attempt

        return best_x, best_f, calls

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x, best_f, calls_used = self._differential_evolution(func, budget, rng)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
