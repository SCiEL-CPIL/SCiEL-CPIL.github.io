"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Differential Evolution optimizer for Büche-Rastrigin f4 on [-5,5]^5.
    
    Uses a large static population (NP=50) with DE/rand/1/bin mutation (F=0.8, CR=0.9)
    to maintain diversity and escape local minima in the rugged multimodal landscape.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        def time_ok():
            return time.time() < deadline

        # DE parameters
        NP = 50  # Population size
        F = 0.8   # Mutation factor
        CR = 0.9  # Crossover rate
        bound_lo = -5.0
        bound_hi = 5.0
        
        # Initialize population uniformly in the box
        pop = rng.uniform(bound_lo, bound_hi, size=(NP, dim))
        pop_f = np.array([func(x) for x in pop])
        budget_used = NP
        
        # Track global best
        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]
        
        # Main DE loop
        while budget_used + NP <= budget and time_ok():
            # Generate trial population using DE/rand/1/bin
            for i in range(NP):
                if budget_used + 1 > budget or not time_ok():
                    break
                
                # Select three distinct random individuals different from i
                while True:
                    r1 = rng.integers(0, NP)
                    if r1 != i:
                        break
                while True:
                    r2 = rng.integers(0, NP)
                    if r2 != i and r2 != r1:
                        break
                while True:
                    r3 = rng.integers(0, NP)
                    if r3 != i and r3 != r1 and r3 != r2:
                        break
                
                # Mutation: v = x[r1] + F * (x[r2] - x[r3])
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Binomial crossover
                j_rand = rng.integers(0, dim)
                trial = np.zeros(dim)
                for j in range(dim):
                    if rng.random() <= CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                # Clip to bounds
                trial = np.clip(trial, bound_lo, bound_hi)
                
                # Evaluate
                f_trial = func(trial)
                budget_used += 1
                
                # Selection: keep better of trial and current
                if f_trial <= pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
        
        # Use remaining budget for local refinement around best point
        if budget_used < budget and time_ok():
            # Simple (1+1)-ES refinement around best
            sigma = 0.3
            x_local = best_x.copy()
            f_local = best_f
            
            while budget_used < budget and time_ok():
                z = rng.standard_normal(dim)
                trial = x_local + sigma * z
                trial = np.clip(trial, bound_lo, bound_hi)
                
                f_trial = func(trial)
                budget_used += 1
                
                if f_trial < f_local:
                    x_local = trial
                    f_local = f_trial
                    if f_local < best_f:
                        best_f = f_local
                        best_x = trial.copy()
                    # Shrink sigma slightly on success
                    sigma *= 0.95
                else:
                    # Grow sigma slightly on failure
                    sigma *= 1.05
                    sigma = min(sigma, 2.0)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
