"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill

        n = self.dim
        budget = self.budget
        
        # DE parameters
        pop_size = 20  # Population size
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate
        best_x = None
        best_f = float('inf')
        
        # Initialize population
        pop = np.random.uniform(-5, 5, (pop_size, n))
        pop_f = np.array([func(x) for x in pop])
        
        # Track best
        idx_best = np.argmin(pop_f)
        best_f = pop_f[idx_best]
        best_x = pop[idx_best].copy()
        
        calls_made = pop_size
        
        # Main DE loop
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Check if we have enough budget for a full generation
            if calls_made + pop_size > budget:
                # Do partial generation
                partial_size = budget - calls_made
            else:
                partial_size = pop_size
            
            # Sort population by fitness
            sort_idx = np.argsort(pop_f)
            pop = pop[sort_idx]
            pop_f = pop_f[sort_idx]
            
            # Update best
            if pop_f[0] < best_f:
                best_f = pop_f[0]
                best_x = pop[0].copy()
            
            # Generate offspring
            for i in range(partial_size):
                if calls_made >= budget:
                    break
                
                # Select three distinct random individuals
                while True:
                    r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                    if r1 != r2 and r2 != r3 and r1 != r3:
                        break
                
                # DE/rand/1/bin mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Binary crossover
                trial = pop[i].copy()
                j_rand = np.random.randint(n)
                for j in range(n):
                    if np.random.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                calls_made += 1
                
                # Selection
                if f_trial <= pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = f_trial
                    
                    # Update global best
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
            
            # Adaptive F and CR based on stagnation
            # Reduce F slightly to converge
            F = max(0.5, F * 0.995)
            CR = min(0.95, CR * 1.001)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
