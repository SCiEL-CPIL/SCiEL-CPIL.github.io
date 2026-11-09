"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution optimizer for BBOB f13."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Differential Evolution Parameters
        # Population size: typically 5-10 * dim for robustness, but we have limited budget.
        # Let's try NP = 2 * dim + 1 = 11 for dim=5? Or maybe a bit larger for global coverage.
        # With budget 500, if NP=20, we can do ~25 generations.
        # If NP=10, we can do ~50 generations.
        # Sharp ridges often benefit from larger populations to avoid getting stuck.
        # Let's use NP = 4 * dim = 20.
        
        NP = 4 * dim
        if NP > 50: # Cap for larger dims if necessary, though dim is fixed at 5 here
            NP = 50
            
        # Scaling factor F and Crossover rate CR
        F = 0.8
        CR = 0.5
        
        rng = np.random.default_rng(12345)
        
        # Initialize population
        # Start with a mix of random points and some structured points to ensure good initial coverage
        pop = rng.uniform(-5, 5, (NP, dim))
        
        # Ensure at least one point is near origin or center, though random is usually fine.
        # For f13 (sharp ridge), the ridge might be anywhere. Random is safer.
        
        # Evaluate initial population
        evals_used = 0
        fitness = np.zeros(NP)
        
        for i in range(NP):
            if evals_used >= budget:
                break
            x = pop[i]
            fx = func(x)
            fitness[i] = fx
            evals_used += 1
            
        # Track best
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]
        
        # Main DE Loop
        # We need to ensure we don't exceed budget.
        # Each generation costs NP evaluations (one per individual).
        # We have budget - evals_used remaining.
        
        while evals_used < budget:
            # Check if we can afford a full generation
            if budget - evals_used < NP:
                # Do partial generation? Or just break.
                # Doing partial generation is messy with DE.
                # Let's just break if we can't do a full generation to maintain algorithm integrity,
                # or do single-step improvements if very tight.
                # Given budget 500 and NP=20, we'll have many generations.
                break
                
            for i in range(NP):
                if evals_used >= budget:
                    break
                    
                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                # Choose distinct indices r1, r2, r3 != i
                r1, r2, r3 = rng.choice(NP, size=3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = rng.choice(NP, size=3, replace=False)
                
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover: Binomial
                trial = np.zeros_like(pop[i])
                j_rand = rng.integers(0, dim)
                
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                        
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate trial
                ft = func(trial)
                evals_used += 1
                
                # Selection
                if ft <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = ft
                    if ft < best_f:
                        best_f = ft
                        best_x = trial.copy()
                        
        return best_x, best_f
# EVOLVE-BLOCK-END
