"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Population-based Differential Evolution (DE) for Rosenbrock rotated."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size: small enough to fit in budget, large enough for diversity
        # With budget=500, dim=5, we can afford a population of ~20-30
        # Let's use pop_size=20, so ~25 generations
        pop_size = 20
        n_evals = 0
        max_evals = budget

        # Initialize population uniformly in [-5, 5]^dim
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fvals = np.empty(pop_size)
        for i in range(pop_size):
            fvals[i] = func(pop[i])
            n_evals += 1
            if n_evals >= max_evals:
                break

        # DE parameters
        F = 0.8  # Scale factor
        CR = 0.9  # Crossover rate
        # Use best/1/DE/rand/1 strategy variants

        while n_evals < max_evals:
            # Sort by fitness
            order = np.argsort(fvals)
            sorted_pop = pop[order]
            sorted_fvals = fvals[order]

            best_x = sorted_pop[0]
            best_f = sorted_fvals[0]

            # Generate offspring for each member
            for i in range(pop_size):
                if n_evals >= max_evals:
                    break

                # Select three distinct random indices different from i
                candidates = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = rng.choice(candidates, size=3, replace=False)

                # Differential evolution: v = x_r1 + F * (x_r2 - x_r3)
                v = sorted_pop[r1] + F * (sorted_pop[r2] - sorted_pop[r3])

                # Also try a variant with the best individual (best/1)
                # v_best = best_x + F * (x_r2 - x_r3)
                # We'll alternate or mix strategies
                if i % 2 == 0:
                    v = best_x + F * (sorted_pop[r2] - sorted_pop[r3])

                # Crossover
                trial = sorted_pop[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]

                # Clip to bounds
                trial = np.clip(trial, -5, 5)

                # Evaluate
                f_trial = func(trial)
                n_evals += 1

                # Selection
                if f_trial < sorted_fvals[i]:
                    sorted_pop[i] = trial
                    sorted_fvals[i] = f_trial

            # Update pop and fvals from sorted versions
            pop = sorted_pop.copy()
            fvals = sorted_fvals.copy()

        # Return the best found
        best_idx = np.argmin(fvals)
        return pop[best_idx], fvals[best_idx]
# EVOLVE-BLOCK-END
