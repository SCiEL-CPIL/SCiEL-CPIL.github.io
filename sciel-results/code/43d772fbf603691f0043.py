"""Rewritten optimizer for BBOB f23 (Katsuura) using population-based DE/rand/1."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution (DE/rand/1) with adaptive F and CR for Katsuura."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        B = self.budget
        deadline = time.time() + 280  # 280s safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Population size: keep it moderate for 5D
        pop_size = max(10, min(30, B // 10))

        # Initialize population uniformly in [-5, 5]^d
        pop = rng.uniform(-5, 5, size=(pop_size, d))
        fvals = np.array([func(x) for x in pop])

        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]

        calls_made = pop_size
        F = 0.8  # mutation factor
        CR = 0.5  # crossover rate

        # Adaptive parameters
        success_count = 0
        trial_count = 0
        sigma = 1.0  # exploration scale

        while calls_made < B:
            if time.time() > deadline:
                break

            # Generate trial vectors for each member of the population
            for i in range(pop_size):
                if calls_made >= B or time.time() > deadline:
                    break

                # Select two distinct random indices different from i
                r1, r2 = rng.choice(pop_size, size=2, replace=False)
                while r1 == i:
                    r1 = rng.choice(pop_size)
                while r2 == i or r2 == r1:
                    r2 = rng.choice(pop_size)

                # Mutation: DE/rand/1
                v = pop[r1] + F * (pop[r2] - pop[i])

                # Crossover (binomial)
                jrand = rng.integers(d)
                cross_mask = rng.random(d) <= CR
                cross_mask[jrand] = True

                u = np.where(cross_mask, v, pop[i])
                u = np.clip(u, -5, 5)

                calls_made += 1
                fu = func(u)

                # Selection
                if fu <= fvals[i]:
                    pop[i] = u
                    fvals[i] = fu
                    if fu < best_f:
                        best_f = fu
                        best_x = u.copy()
                    success_count += 1

                trial_count += 1

            # Adaptive F: adjust based on recent success rate
            if trial_count > 0:
                success_rate = success_count / trial_count
                # Increase F if success rate is low (more exploration), decrease if high
                F = np.clip(0.4 + 0.4 * (1.0 - success_rate), 0.3, 1.0)
                CR = np.clip(0.3 + 0.4 * (1.0 - success_rate), 0.2, 0.9)

            # Reset counters for next generation
            success_count = 0
            trial_count = 0

            # If population has converged (low variance), perturb
            if np.std(fvals) < 1e-10:
                # Restart one random member
                idx = rng.integers(pop_size)
                pop[idx] = rng.uniform(-5, 5, size=d)
                calls_made += 1
                fvals[idx] = func(pop[idx])
                if fvals[idx] < best_f:
                    best_f = fvals[idx]
                    best_x = pop[idx].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
