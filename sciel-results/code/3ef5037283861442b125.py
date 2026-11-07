"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based Differential Evolution with deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        
        # Initialize population (lambda = 5)
        pop = rng.uniform(-5, 5, (5, dim))
        fpop = np.array([func(x) for x in pop])
        
        best_idx = np.argmin(fpop)
        best_x = pop[best_idx].copy()
        best_f = fpop[best_idx]
        
        # DE Parameters
        F = 0.8  # Differential weight
        CR = 0.5  # Crossover rate
        
        calls_made = 5
        
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            # Select reference individuals
            # best/2/bin strategy: x_best, x_rand1, x_rand2
            # Ensure indices are distinct
            candidates = [i for i in range(5) if i != best_idx]
            r1 = rng.choice(candidates)
            r2 = rng.choice([i for i in candidates if i != r1])
            
            x_best = pop[best_idx]
            x_r1 = pop[r1]
            x_r2 = pop[r2]
            
            # Mutation: v = x_best + F * (x_r1 - x_r2)
            v = x_best + F * (x_r1 - x_r2)
            
            # Clip to bounds
            v = np.clip(v, -5, 5)
            
            # Crossover (binomial)
            j_rand = rng.integers(0, dim)
            trial = np.zeros(dim)
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = pop[best_idx][j]
                    
            # Evaluate
            f_trial = func(trial)
            calls_made += 1
            
            # Selection
            if f_trial < fpop[best_idx]:
                pop[best_idx] = trial
                fpop[best_idx] = f_trial
                best_x = trial.copy()
                best_f = f_trial
            else:
                # If no improvement, we can still update the population 
                # to maintain diversity if the trial is better than the 
                # individual it replaced, but in best/1 we replace best.
                # Standard DE replaces the target if trial is better.
                # Here target is always best_idx.
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
