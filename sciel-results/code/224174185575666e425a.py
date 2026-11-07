"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Differential Evolution for rotated Weierstrass landscape."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # DE Parameters
        # Population size usually 5 * dim for good performance
        N = 5 * dim 
        if N < 10:
            N = 10
            
        # Ensure we don't exceed budget with initial population + generations
        # Initial population takes N calls.
        # Remaining budget for generations: (budget - N)
        # Each generation takes N calls.
        
        # Initialize population
        # Weierstrass is often minimized near 0, but shifted/rotated.
        # Uniform initialization in [-5, 5] is safe.
        pop = rng.uniform(-5, 5, size=(N, dim))
        
        # Evaluate initial population
        fitness = np.array([func(np.clip(x, -5, 5)) for x in pop])
        
        calls_made = N
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]
        
        # DE parameters
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover rate
        
        # We have budget - N calls left
        # Each generation consumes N calls
        # We can run for approximately (budget - N) / N generations
        # But we should stop if time runs out.
        
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Check if we can afford another generation
            if calls_made + N > budget:
                # If we can't afford a full generation, we might do partial or break.
                # Standard DE requires full population update.
                # We can break here to be safe, or handle remainder.
                # Given budget=500 and N=25, 500 is divisible by 25?
                # 500 / 25 = 20 generations.
                # So it fits perfectly if N divides budget.
                # If not, we break.
                break
            
            # Update parameters slightly to balance exploration/exploitation
            # F can be kept constant or decayed. CR can be kept constant.
            # Let's try a slight decay of F to exploit more as we progress?
            # Or keep F constant. Standard DE works well with constant F.
            
            # Generate new population
            new_pop = np.zeros((N, dim))
            new_fitness = np.full(N, np.inf)
            
            for i in range(N):
                # Select three distinct random indices different from i
                r1, r2, r3 = rng.choice(N, size=3, replace=False)
                while r1 == i:
                    r1 = rng.integers(0, N)
                while r2 == i or r2 == r1:
                    r2 = rng.integers(0, N)
                while r3 == i or r3 == r1 or r3 == r2:
                    r3 = rng.integers(0, N)
                    
                # Mutation
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover (Binomial)
                # Random vector for crossover
                j_rand = rng.integers(0, dim)
                cross_mask = rng.random(dim) < CR
                cross_mask[j_rand] = True  # Ensure at least one gene is from mutant
                
                # Create trial vector
                trial = np.where(cross_mask, mutant, pop[i])
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                calls_made += 1
                
                # Selection
                if f_trial <= fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = f_trial
                    # Update global best
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                else:
                    new_pop[i] = pop[i]
                    new_fitness[i] = fitness[i]
            
            # Replace population
            pop = new_pop
            fitness = new_fitness
            
        return best_x, best_f
# EVOLVE-BLOCK-END
