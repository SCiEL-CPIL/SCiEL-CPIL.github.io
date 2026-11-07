"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution for continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size
        npop = max(10, 5 * dim)
        
        # Initialize population uniformly in [-5, 5]
        pop = rng.uniform(-5, 5, size=(npop, dim))
        fvals = np.array([func(x) for x in pop])
        
        calls = npop
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_fx = fvals[best_idx]

        # DE parameters
        F = 0.8  # Scale factor
        CR = 0.9  # Crossover rate

        while calls < budget:
            for i in range(npop):
                if calls >= budget:
                    break
                
                # Select three distinct indices different from i
                # Use a simple loop to ensure distinctness and != i
                r1, r2, r3 = rng.choice(npop, 3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = rng.choice(npop, 3, replace=False)

                # Generate trial vector
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                trial = pop[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Boundary handling: clip
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                calls += 1
                
                # Selection
                if f_trial <= fvals[i]:
                    pop[i] = trial
                    fvals[i] = f_trial
                    if f_trial < best_fx:
                        best_fx = f_trial
                        best_x = trial.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
