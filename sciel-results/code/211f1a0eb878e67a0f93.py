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

        pop_size = 20
        n_evals = 0
        max_evals = budget

        # Initialize population: mix of uniform and concentrated near origin
        pop = np.empty((pop_size, dim))
        for i in range(pop_size):
            if i < pop_size // 2:
                pop[i] = rng.uniform(-5, 5, dim)
            else:
                pop[i] = rng.uniform(-2, 2, dim)

        fvals = np.empty(pop_size)
        for i in range(pop_size):
            fvals[i] = func(pop[i])
            n_evals += 1
            if n_evals >= max_evals:
                break

        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]

        # DE parameters
        F_start = 0.8
        F_end = 0.4
        CR = 0.7

        while n_evals < max_evals:
            progress = n_evals / max_evals
            F = F_start + (F_end - F_start) * progress

            # Generate offspring for each member
            for i in range(pop_size):
                if n_evals >= max_evals:
                    break

                # DE/best/1 variant:
                # v = best + F*(x_r1 - x_r2)
                mask = np.ones(pop_size, dtype=bool)
                mask[i] = False
                mask[best_idx] = False
                candidates = np.where(mask)[0]

                if len(candidates) >= 3:
                    r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
                    v = best_x + F * (pop[r1] - pop[r2]) + 0.5 * F * (pop[r3] - pop[r2])
                elif len(candidates) >= 2:
                    r1, r2 = rng.choice(candidates, size=2, replace=False)
                    v = best_x + F * (pop[r1] - pop[r2])
                else:
                    v = pop[i] + rng.normal(0, 0.5, dim)

                # Reflect back into bounds
                for j in range(dim):
                    if v[j] < -5:
                        v[j] = -5 + (-5 - v[j]) * 0.5
                    elif v[j] > 5:
                        v[j] = 5 - (v[j] - 5) * 0.5
                    v[j] = np.clip(v[j], -5, 5)

                # Binomial crossover
                trial = pop[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]

                # Evaluate
                f_trial = func(trial)
                n_evals += 1

                # Selection: keep better
                if f_trial < fvals[i]:
                    pop[i] = trial
                    fvals[i] = f_trial

                # Update global best
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()
                    best_idx = i

            # Update best after full generation
            best_idx = np.argmin(fvals)
            best_x = pop[best_idx].copy()
            best_f = fvals[best_idx]

        return pop[best_idx], fvals[best_idx]
# EVOLVE-BLOCK-END
