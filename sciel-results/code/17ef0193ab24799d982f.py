"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution optimizer for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        # DE Parameters
        NP = 10  # Population size
        F = 0.8  # Mutation factor
        CR = 0.5 # Crossover rate
        
        # Initialize population
        pop = rng.uniform(-5, 5, size=(NP, n))
        fvals = np.array([func(x) for x in pop])
        
        calls_made = NP
        
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]
        
        # Main loop
        while calls_made < budget:
            # Optional: Shuffle population indices to process
            # But we iterate 0..NP-1, so we can just process in order.
            # To avoid bias, we can permute the order of processing.
            order = rng.permutation(NP)
            
            for i in order:
                if calls_made >= budget:
                    break
                
                # Select three distinct random indices r1, r2, r3 != i
                r1 = rng.integers(0, NP)
                while r1 == i:
                    r1 = rng.integers(0, NP)
                
                r2 = rng.integers(0, NP)
                while r2 == i or r2 == r1:
                    r2 = rng.integers(0, NP)
                    
                r3 = rng.integers(0, NP)
                while r3 == i or r3 == r1 or r3 == r2:
                    r3 = rng.integers(0, NP)
                
                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover (Binomial)
                j_rand = rng.integers(0, n)
                mask = rng.random(n) < CR
                mask[j_rand] = True
                u = np.where(mask, v, pop[i])
                
                # Clip to bounds
                u = np.clip(u, -5, 5)
                
                # Evaluation
                f_u = func(u)
                calls_made += 1
                
                # Selection
                if f_u < fvals[i]:
                    pop[i] = u
                    fvals[i] = f_u
                    if f_u < best_f:
                        best_f = f_u
                        best_x = u.copy()
                        
        return best_x, best_f
# EVOLVE-BLOCK-END
