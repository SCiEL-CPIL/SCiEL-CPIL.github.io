"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution optimizer for BBOB-style Weierstrass function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Differential Evolution parameters
        # Budget 500, Dim 5.
        # Population size NP.
        # Init cost: NP calls.
        # Each generation cost: NP calls.
        # Total calls: NP + G * NP <= budget.
        # Let's try NP = 15.
        # Init: 15. Remaining: 485.
        # Gens: 485 // 15 = 32. Total: 15 + 32*15 = 495.
        # This leaves 5 calls buffer.
        
        NP = 15
        F = 0.8  # Scaling factor
        CR = 0.5 # Crossover rate
        
        # Initialize population
        pop = rng.uniform(-5, 5, (NP, dim))
        fvals = np.empty(NP)
        total_calls = 0
        global_best_f = np.inf
        global_best_x = np.zeros(dim)
        
        for i in range(NP):
            fvals[i] = func(pop[i])
            total_calls += 1
            if fvals[i] < global_best_f:
                global_best_f = fvals[i]
                global_best_x = pop[i].copy()
            if total_calls >= budget:
                return global_best_x, global_best_f
        
        # Main loop
        # We calculate max generations based on remaining budget
        remaining = budget - total_calls
        max_gens = remaining // NP
        
        for gen in range(max_gens):
            # Update best index
            best_idx = np.argmin(fvals)
            
            for i in range(NP):
                if total_calls >= budget:
                    return global_best_x, global_best_f
                
                # Select 3 distinct random indices different from i
                r1, r2, r3 = rng.choice(NP, size=3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    # Ensure distinct and not i
                    # Simple retry loop, unlikely to be infinite
                    r1, r2, r3 = rng.choice(NP, size=3, replace=False)
                    if r1 == i or r2 == i or r3 == i:
                         # Force distinct
                         candidates = np.setdiff1d(np.arange(NP), [i])
                         r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
                         break
                
                # Mutation
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover (Binary)
                trial = pop[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                total_calls += 1
                
                # Selection
                if f_trial <= fvals[i]:
                    pop[i] = trial
                    fvals[i] = f_trial
                    if f_trial < global_best_f:
                        global_best_f = f_trial
                        global_best_x = trial.copy()
        
        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
