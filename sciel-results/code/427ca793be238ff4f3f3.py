"""Rewritten optimizer using Differential Evolution for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution optimizer for BBOB-style f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Population size: use 40 individuals, leaving room for multiple generations
        pop_size = 40
        F = 0.8  # mutation factor
        CR = 0.65  # crossover rate

        # Initialize population uniformly in [-5, 5]^dim
        pop = rng.uniform(-5, 5, size=(pop_size, dim))
        evals = np.zeros(pop_size)
        for i in range(pop_size):
            evals[i] = func(pop[i])

        best_idx = np.argmin(evals)
        best_val = evals[best_idx]
        best_x = pop[best_idx].copy()
        calls_made = pop_size

        # Track history for AOCC
        history = []
        history.append(best_val)

        # Adaptive F: decrease from 0.85 to 0.5 over generations
        total_generations = (budget - pop_size) // pop_size
        gen = 0

        while calls_made < budget:
            # Check time
            if time.time() > deadline:
                break

            # Adaptive F
            if total_generations > 0:
                progress = gen / total_generations
                F_adapt = 0.85 - 0.35 * progress
            else:
                F_adapt = 0.8

            # Generate offspring for each individual
            for i in range(pop_size):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                # Select two distinct random indices different from i
                r1, r2 = rng.choice(pop_size, size=2, replace=False)
                while r1 == i:
                    r1 = rng.integers(pop_size)
                while r2 == i or r2 == r1:
                    r2 = rng.integers(pop_size)

                # DE/rand/1 mutation
                mutant = pop[r1] + F_adapt * (pop[r2] - pop[i])

                # Binary crossover
                cross_mask = rng.random(dim) < CR
                if not np.any(cross_mask):
                    cross_mask[rng.integers(dim)] = True

                trial = np.where(cross_mask, mutant, pop[i])
                trial = np.clip(trial, -5, 5)

                trial_val = func(trial)
                calls_made += 1

                # Selection
                if trial_val <= evals[i]:
                    pop[i] = trial
                    evals[i] = trial_val
                    if trial_val < best_val:
                        best_val = trial_val
                        best_x = trial.copy()

            gen += 1
            history.append(best_val)

        # Fill remaining history entries if needed
        while len(history) < budget:
            history.append(best_val)

        return best_x, best_val
# EVOLVE-BLOCK-END
