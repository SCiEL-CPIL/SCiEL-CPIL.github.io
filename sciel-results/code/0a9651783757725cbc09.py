"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution (DE) with adaptive parameters for Rastrigin-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 300s limit

        # Population size
        pop_size = 20
        # Initial F and CR
        F = 0.8
        CR = 0.5

        # Initialize population uniformly in [-5, 5]^dim
        rng = np.random.default_rng(42)
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fit = np.array([func(ind) for ind in pop])

        best_idx = np.argmin(fit)
        best_x = pop[best_idx].copy()
        best_f = fit[best_idx]

        # Track for adaptive F
        success_count = 0
        total_mutations = 0
        F_adapt = F
        CR_adapt = CR

        eval_count = pop_size

        # DE loop
        while eval_count < budget:
            # Check time
            if time.time() > deadline - 1.0:
                break

            # Adaptive parameter update every 10 generations
            if total_mutations > 0 and total_mutations % 10 == 0:
                # Adjust F based on success rate
                if success_count > 0:
                    F_adapt = np.clip(F_adapt + 0.1, 0.3, 1.0)
                else:
                    F_adapt = np.clip(F_adapt - 0.1, 0.3, 1.0)
                success_count = 0
                # Slight CR adjustment
                CR_adapt = np.clip(CR_adapt + rng.normal(0, 0.05), 0.1, 0.9)

            # Generate offspring for each individual
            for i in range(pop_size):
                if eval_count >= budget:
                    break
                if time.time() > deadline - 0.5:
                    break

                # Select three distinct random indices
                r1, r2, r3 = np.random.choice(
                    [j for j in range(pop_size) if j != i], 3, replace=False
                )

                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                v = pop[r1] + F_adapt * (pop[r2] - pop[r3])

                # Crossover
                j_rand = rng.integers(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.random() < CR_adapt or j == j_rand:
                        trial[j] = v[j]

                # Clip to bounds
                trial = np.clip(trial, -5, 5)

                # Evaluate
                trial_f = func(trial)
                eval_count += 1

                total_mutations += 1

                # Selection
                if trial_f <= fit[i]:
                    pop[i] = trial
                    fit[i] = trial_f
                    success_count += 1

                    if trial_f < best_f:
                        best_f = trial_f
                        best_x = trial.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
