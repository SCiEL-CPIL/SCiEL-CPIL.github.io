"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution optimizer for the BBOB f24 task."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # DE parameters
        NP = 5 * dim  # Population size
        F = 0.5  # Mutation factor
        CR = 0.9  # Crossover rate

        # Initialize population
        pop = rng.uniform(-5, 5, (NP, dim))
        fit = np.array([func(x) for x in pop])
        calls_used = NP

        best_idx = np.argmin(fit)
        best_x = pop[best_idx].copy()
        best_f = fit[best_idx]

        # DE loop
        while calls_used < budget:
            for i in range(NP):
                if calls_used >= budget:
                    break

                # Select 3 distinct random indices different from i
                r1, r2, r3 = rng.choice([j for j in range(NP) if j != i], 3, replace=False)

                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])

                # Crossover (binomial)
                jrand = rng.integers(0, dim)
                u = rng.random(dim)
                c = np.where(u < CR, v, pop[i])
                c[jrand] = v[jrand]

                # Clip to bounds
                c = np.clip(c, -5, 5)

                # Evaluate
                fc = func(c)
                calls_used += 1

                # Selection
                if fc <= fit[i]:
                    pop[i] = c
                    fit[i] = fc
                    if fc < best_f:
                        best_f = fc
                        best_x = c.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
