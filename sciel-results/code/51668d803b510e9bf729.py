# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution (DE/rand/1) for multi-objective optimization (DTLZ4)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Initial population size
        pop_size = 40
        if self.budget < pop_size * 2:
            pop_size = max(10, self.budget // 4)

        # Initialize population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Archive of non-dominated points (for tracking, but we keep full population)
        # DE/rand/1/
        F_factor = 0.8
        CR = 0.9

        while used < self.budget:
            # Select three distinct random parents
            i = rng.integers(pop_size)
            j1 = rng.integers(pop_size)
            j2 = rng.integers(pop_size)
            j3 = rng.integers(pop_size)
            # Ensure distinctness (simple retry logic)
            while j1 == i or j2 == i or j3 == i or j1 == j2 or j1 == j3 or j2 == j3:
                j1 = rng.integers(pop_size)
                j2 = rng.integers(pop_size)
                j3 = rng.integers(pop_size)

            # DE mutation: v = x_j1 + F * (x_j2 - x_j3)
            v = X[j1] + F_factor * (X[j2] - X[j3])
            
            # Clip to bounds
            v = np.clip(v, lo, hi)

            # Crossover
            trial = X[i].copy()
            rand_idx = rng.integers(self.dim)
            for k in range(self.dim):
                if rng.random() < CR or k == rand_idx:
                    trial[k] = v[k]
            
            # Evaluate trial
            f_trial = problem(trial)
            used += 1

            # Selection: keep if non-dominated by X[i]
            # Check if trial dominates X[i]
            dom_trial_over_i = np.all(f_trial <= F[i]) and np.any(f_trial < F[i])
            # Check if X[i] dominates trial
            dom_i_over_trial = np.all(F[i] <= f_trial) and np.any(F[i] < f_trial)
            
            if dom_trial_over_i or not dom_i_over_trial:
                # Trial is better or non-dominated, accept
                X[i] = trial
                F[i] = f_trial

        return X
# EVOLVE-BLOCK-END
