"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Population-based Differential Evolution (DE/rand/1/bin) with adaptive
    scaling and crossover, plus a local refinement phase using CMA-like
    step-size adaptation on the best individual.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # 280s safety margin within 300s limit

        rng = np.random.default_rng(42)

        # Population size
        pop_size = 20
        if pop_size > budget:
            pop_size = max(2, budget)

        # Initialize population uniformly in [-5, 5]^dim
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fitness = np.array([func(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]

        calls_used = pop_size
        remaining = budget - calls_used

        # DE parameters
        F = 0.8  # scaling factor
        CR = 0.9  # crossover rate

        # Track best for early stopping
        best_f_history = [best_f]

        while remaining > 0 and time.time() < deadline:
            # Check for timeout
            if time.time() > deadline - 5:
                break

            # Select random distinct indices for DE/rand/1/bin
            i = rng.integers(0, pop_size)
            a, b, c = rng.choice(pop_size, size=3, replace=False)
            while a == i:
                a = rng.integers(0, pop_size)
            while b == i or b == a:
                b = rng.integers(0, pop_size)
            while c == i or c == a or c == b:
                c = rng.integers(0, pop_size)

            # Mutation: v = x_a + F * (x_b - x_c)
            v = pop[a] + F * (pop[b] - pop[c])
            v = np.clip(v, -5, 5)

            # Binomial crossover
            j_rand = rng.integers(0, dim)
            trial = np.copy(pop[i])
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    trial[j] = v[j]

            # Evaluate trial
            trial_f = func(trial)
            calls_used += 1
            remaining = budget - calls_used

            # Selection
            if trial_f < fitness[i]:
                pop[i] = trial
                fitness[i] = trial_f
                if trial_f < best_f:
                    best_f = trial_f
                    best_x = trial.copy()
                    best_f_history.append(best_f)

        # Local refinement: use remaining budget for a (1+1)-ES on best_x
        if remaining > 0 and time.time() < deadline - 2:
            sigma = 0.5
            x = best_x.copy()
            fx = best_f
            for _ in range(remaining):
                if time.time() > deadline - 2:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
