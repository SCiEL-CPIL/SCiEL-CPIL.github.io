"""Differential Evolution with adaptive parameters (jDE) for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derivative-free optimizer using Differential Evolution with adaptive control."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        deadline = time.time() + 280  # safety margin before 300s limit

        # Population size
        lam = 20
        dim = self.dim

        # Initialize population uniformly in [-5, 5]^dim
        pop = rng.uniform(-5, 5, (lam, dim))
        fvals = np.array([func(x) for x in pop])

        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]

        # Adaptive parameters
        F = 0.7
        CR = 0.5
        F_target = 0.8
        CR_target = 0.3
        tau_F = 0.1
        tau_CR = 0.1

        calls_used = lam
        remaining = self.budget - lam

        while remaining > 0:
            if time.time() > deadline:
                break

            # Sort population by fitness
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]

            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = pop[0].copy()

            # Generate offspring
            new_pop = np.empty_like(pop)
            new_fvals = np.empty(lam)
            new_pop[0] = pop[0]  # Elitism: keep best
            new_fvals[0] = fvals[0]

            for i in range(1, lam):
                # Select three distinct random individuals different from i
                while True:
                    r1, r2, r3 = rng.choice(lam, 3, replace=False)
                    if len({r1, r2, r3, i}) == 4:
                        break

                # Differential mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])

                # Crossover
                j_rand = rng.integers(dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]

                # Clip to bounds
                trial = np.clip(trial, -5, 5)

                # Evaluate
                calls_used += 1
                remaining -= 1
                if remaining < 0:
                    break
                ft = func(trial)
                new_fvals[i] = ft

                # Selection
                if ft <= fvals[i]:
                    new_pop[i] = trial
                else:
                    new_pop[i] = pop[i]

            if remaining < 0:
                break

            pop = new_pop
            fvals = new_fvals

            # Update best
            best_idx = np.argmin(fvals)
            if fvals[best_idx] < best_f:
                best_f = fvals[best_idx]
                best_x = pop[best_idx].copy()

            # Adaptive parameter control (jDE)
            if rng.random() < 0.1:
                F = rng.uniform(0, 1)
            F = np.clip(F + rng.normal(0, tau_F), 0.1, 1.0)

            if rng.random() < 0.1:
                CR = rng.uniform(0, 1)
            CR = np.clip(CR + rng.normal(0, tau_CR), 0.0, 1.0)

            # Gradually shift towards target values
            F = F + 0.01 * (F_target - F)
            CR = CR + 0.01 * (CR_target - CR)

        return best_x, best_f
# EVOLVE-BLOCK-END
