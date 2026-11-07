"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start DE/rand/1 with population-based diversity maintenance."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 290  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size: use a moderate population
        pop_size = 50
        # Ensure we don't exceed budget
        if pop_size >= budget:
            pop_size = max(10, budget // 10)

        # Stratified initialization: divide [-5,5] into sub-regions
        # Use Latin Hypercube-like stratification
        x = np.zeros((pop_size, dim))
        for i in range(pop_size):
            for d in range(dim):
                # Stratify each dimension
                lo = -5 + (10.0 * i) / pop_size
                hi = -5 + (10.0 * (i + 1)) / pop_size
                x[i, d] = rng.uniform(lo, hi)
        # Clip to be safe
        x = np.clip(x, -5, 5)

        # Evaluate initial population
        f = np.zeros(pop_size)
        for i in range(pop_size):
            f[i] = func(x[i])
            if time.time() > deadline:
                break
            # Track calls
            calls_made = i + 1
            if calls_made >= budget:
                break

        # If we ran out of budget during initialization, return best
        if calls_made >= budget:
            idx = np.argmin(f[:calls_made])
            return x[idx], f[idx]

        # Track best
        best_idx = np.argmin(f[:calls_made])
        best_x = x[best_idx].copy()
        best_f = f[best_idx]

        # DE parameters
        F = 0.8  # scale factor
        CR = 0.9  # crossover rate

        # Adaptive: periodically re-inject diversity
        reinject_interval = max(20, pop_size)

        # Main loop
        while calls_made < budget and time.time() < deadline:
            # Select three distinct individuals (DE/rand/1)
            r1 = rng.integers(0, calls_made)
            r2 = rng.integers(0, calls_made)
            r3 = rng.integers(0, calls_made)
            # Ensure distinct
            while r2 == r1:
                r2 = rng.integers(0, calls_made)
            while r3 == r1 or r3 == r2:
                r3 = rng.integers(0, calls_made)

            # Don't mutate the current best if possible (preserve diversity)
            target_idx = rng.integers(0, calls_made)
            if target_idx == best_idx:
                target_idx = (target_idx + 1) % calls_made

            # Differential mutation
            v = x[target_idx] + F * (x[r1] - x[r2] + x[r3] - x[target_idx])

            # Binary crossover
            trial = x[target_idx].copy()
            jrand = rng.integers(0, dim)
            for j in range(dim):
                if rng.random() < CR or j == jrand:
                    trial[j] = v[j]
            trial = np.clip(trial, -5, 5)

            # Evaluate trial
            ft = func(trial)
            calls_made += 1

            if ft < f[target_idx]:
                x[target_idx] = trial
                f[target_idx] = ft
                if ft < best_f:
                    best_f = ft
                    best_x = trial.copy()
                    best_idx = target_idx
            elif ft < best_f:
                best_f = ft
                best_x = trial.copy()
                best_idx = target_idx

            # Periodic re-injection of diversity
            if calls_made % reinject_interval == 0:
                # Replace the worst individual with a new random point
                worst_idx = np.argmax(f[:calls_made])
                if worst_idx != best_idx:
                    x[worst_idx] = rng.uniform(-5, 5, dim)
                    x[worst_idx] = np.clip(x[worst_idx], -5, 5)
                    f[worst_idx] = func(x[worst_idx])
                    calls_made += 1
                    if f[worst_idx] < best_f:
                        best_f = f[worst_idx]
                        best_x = x[worst_idx].copy()
                        best_idx = worst_idx

            if calls_made >= budget:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
