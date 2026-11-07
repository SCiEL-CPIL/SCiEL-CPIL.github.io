"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution (DE/rand/1/bin) with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Population size
        pop_size = 10
        
        # Initialize population uniformly in [-5, 5]^dim
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fvals = np.array([func(x) for x in pop])
        calls = pop_size
        
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]
        
        # DE Parameters
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover rate
        
        while calls < budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Select base individual (randomly)
            r1 = rng.integers(0, pop_size)
            
            # Select two distinct individuals for mutation
            candidates = list(range(pop_size))
            candidates.remove(r1)
            r2, r3 = rng.choice(candidates, 2, replace=False)
            
            # Mutation: v = x_r1 + F * (x_r2 - x_r3)
            v = pop[r1] + F * (pop[r2] - pop[r3])
            v = np.clip(v, -5, 5)
            
            # Crossover: binomial
            j_rand = rng.integers(0, dim)
            trial = pop[r1].copy()
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    trial[j] = v[j]
            
            trial = np.clip(trial, -5, 5)
            
            # Evaluation
            ft = func(trial)
            calls += 1
            
            # Selection
            if ft <= fvals[r1]:
                pop[r1] = trial
                fvals[r1] = ft
                if ft < best_f:
                    best_f = ft
                    best_x = trial.copy()
            
            # Periodic restart if stuck (optional, but DE usually handles this)
            # For bent cigar, maybe increase F if no improvement for a while?
            # Keep it simple for now.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
