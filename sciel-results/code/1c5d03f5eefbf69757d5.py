"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer combining LHS init, adaptive mutation, and deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # LHS-inspired initialization for better spread
        pop_size = int(min(20, self.budget // 4))
        X = np.zeros((pop_size, self.dim))
        for i in range(pop_size):
            for j in range(self.dim):
                X[i, j] = lo[j] + (i + rng.random()) / pop_size * span[j]
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Main loop with deadline check
        while used < self.budget:
            if time.time() > deadline:
                break

            # Find non-dominated set
            dominated = np.array([
                np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1))
                for f in F
            ])
            parents = X[~dominated] if np.any(~dominated) else X

            # Select parent with tournament-like selection from non-dominated
            parent = parents[rng.integers(len(parents))]

            # Adaptive mutation: start large, decrease over time
            progress = used / self.budget
            mut_scale = 0.2 * (1.0 - 0.5 * progress)
            child = parent + mut_scale * span * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace worst point (by sum of objectives)
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst], F[worst] = child, fc

            # Occasionally do a random restart if stuck
            if used % 50 == 0 and used < self.budget - 10:
                rand_idx = rng.integers(len(X))
                X[rand_idx] = lo + span * rng.random(self.dim)
                F[rand_idx] = problem(X[rand_idx])
                used += 1

        return X
# EVOLVE-BLOCK-END
