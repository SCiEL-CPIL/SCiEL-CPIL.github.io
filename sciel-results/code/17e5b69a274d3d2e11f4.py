"""Rewritten optimizer for BBOB f16 (Weierstrass) using DE with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with adaptive parameters for Weierstrass."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Population size
        pop_size = 20
        # DE parameters
        F = 0.8
        CR = 0.5
        # Adaptive sigma for local refinement
        sigma = 1.0

        # Initialize population
        pop = np.random.uniform(-5, 5, (pop_size, dim))
        # Add a few well-spaced points for better coverage
        pop[0] = np.zeros(dim)
        pop[1] = np.array([5.0] * dim)
        pop[2] = np.array([-5.0] * dim)
        pop[3] = np.array([5.0, -5.0, 5.0, -5.0, 5.0])
        pop[4] = np.array([-5.0, 5.0, -5.0, 5.0, -5.0])

        fvals = np.array([func(x) for x in pop])
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]
        calls_made = pop_size

        # Main DE loop
        gen = 0
        while calls_made < budget:
            # Check time
            if time.time() > deadline:
                break

            gen += 1

            # Adaptive F and CR
            F_adapt = max(0.5, F - 0.3 * gen / 50.0)
            CR_adapt = min(0.9, CR + 0.1 * gen / 50.0)

            # Sort population by fitness
            sort_idx = np.argsort(fvals)
            pop_sorted = pop[sort_idx]
            fvals_sorted = fvals[sort_idx]

            for i in range(pop_size):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                # DE/rand/1 mutation
                a, b, c = [j for j in range(pop_size) if j != i]
                a = rng.choice(a, size=3, replace=False)
                x_r1, x_r2, x_r3 = pop_sorted[a[0]], pop_sorted[a[1]], pop_sorted[a[2]]
                x_base = pop_sorted[i]

                v = x_base + F_adapt * (x_r1 - x_r2) + 0.5 * rng.standard_normal(dim) * sigma

                # Crossover
                j_rand = rng.integers(dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.random() < CR_adapt or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = x_base[j]

                trial = np.clip(trial, -5, 5)
                f_trial = func(trial)
                calls_made += 1

                # Selection
                if f_trial < fvals_sorted[i]:
                    pop_sorted[i] = trial
                    fvals_sorted[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()

            # Update pop and fvals
            pop = pop_sorted[sort_idx.argsort()]
            fvals = fvals_sorted[sort_idx.argsort()]

            # Update sigma based on progress
            if gen % 5 == 0:
                sigma *= 0.9

        # Final local refinement around best_x
        if calls_made < budget:
            local_sigma = max(0.01, sigma * 0.5)
            for _ in range(min(50, budget - calls_made)):
                if time.time() > deadline:
                    break
                x_trial = best_x + local_sigma * rng.standard_normal(dim)
                x_trial = np.clip(x_trial, -5, 5)
                f_trial = func(x_trial)
                calls_made += 1
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = x_trial.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
