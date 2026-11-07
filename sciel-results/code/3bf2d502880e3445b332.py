"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution for Büche-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0
        rng = np.random.default_rng(0)

        pop_size = 25
        F = 0.8
        CR = 0.5

        # Initialize population
        pop = rng.uniform(-5, 5, (pop_size, dim))
        pop_f = np.array([func(x) for x in pop])
        budget_used = pop_size

        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]

        while budget_used < budget:
            if time.time() > deadline:
                break

            # For each individual, generate a mutant
            for i in range(pop_size):
                if budget_used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Select three distinct random indices different from i
                a, b, c = rng.choice([j for j in range(pop_size) if j != i], 3, replace=False)

                # DE/rand/1 mutation
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, -5, 5)

                # Crossover
                trial = pop[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]

                # Evaluate trial
                trial_f = func(trial)
                budget_used += 1

                # Selection
                if trial_f < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = trial_f
                    if trial_f < best_f:
                        best_f = trial_f
                        best_x = trial.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
