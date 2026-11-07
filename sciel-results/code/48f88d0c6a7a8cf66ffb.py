"""Optimized black-box optimizer for BBOB f3 (Rastrigin) on [-5,5]^5."""
import numpy as np
import time


class Optimizer:
    """Differential Evolution with adaptive parameters and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initialize population
        pop_size = 20
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fpop = np.array([func(x) for x in pop])

        best_idx = np.argmin(fpop)
        best_x = pop[best_idx].copy()
        best_f = fpop[best_idx]

        # Adaptive parameters
        F = 0.8  # differential weight
        CR = 0.5  # crossover rate

        eval_count = pop_size
        no_improve = 0

        for gen in range(1, 1000):
            if eval_count >= budget:
                break
            if time.time() > deadline:
                break

            # Update best
            for i in range(pop_size):
                if fpop[i] < best_f:
                    best_f = fpop[i]
                    best_x = pop[i].copy()

            # Adaptive F and CR based on stagnation
            if no_improve > 10:
                F = min(1.0, F + 0.05)
                CR = min(0.9, CR + 0.05)
            elif no_improve > 20:
                # Reset population partially
                for i in range(pop_size // 2):
                    idx = rng.integers(pop_size)
                    pop[i] = best_x + rng.standard_normal(dim) * 2.0
                    pop[i] = np.clip(pop[i], -5, 5)
                    fpop[i] = func(pop[i])
                    eval_count += 1
                    if eval_count >= budget or time.time() > deadline:
                        return best_x, best_f
                no_improve = 0

            # Generate offspring
            new_pop = pop.copy()
            new_fpop = fpop.copy()

            for i in range(pop_size):
                if eval_count >= budget or time.time() > deadline:
                    break

                # Select three distinct indices different from i
                while True:
                    r1 = rng.integers(pop_size)
                    if r1 != i:
                        break
                while True:
                    r2 = rng.integers(pop_size)
                    if r2 != i and r2 != r1:
                        break
                while True:
                    r3 = rng.integers(pop_size)
                    if r3 != i and r3 != r1 and r3 != r2:
                        break

                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])

                # Crossover
                jrand = rng.integers(dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.random() < CR or j == jrand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]

                trial = np.clip(trial, -5, 5)
                ft = func(trial)
                eval_count += 1

                # Selection
                if ft <= fpop[i]:
                    new_pop[i] = trial
                    new_fpop[i] = ft

            pop = new_pop
            fpop = new_fpop

            # Check for improvement
            gen_best = np.min(fpop)
            if gen_best < best_f - 1e-10:
                no_improve = 0
            else:
                no_improve += 1

        return best_x, best_f
