"""Multi-objective optimizer on WFG7: Differential Evolution (DE/rand/1) with time-deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """DE-based multi-objective optimizer with wall-clock deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # 10s safety margin

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(42)

        # Population size: large enough for diversity, small enough for speed
        pop_size = min(40, max(20, self.budget // 5))

        # Initialize population uniformly
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Track all evaluated objective vectors for hypervolume
        all_F = F.copy()

        # DE parameters
        CR = 0.9  # crossover rate
        F_scale = 0.8  # differential weight

        while used < self.budget:
            if time.time() > deadline:
                break

            # For each individual, create one trial vector via DE/rand/1
            # We'll batch-process to stay within budget
            n_trials = min(pop_size, self.budget - used)
            if n_trials <= 0:
                break

            # Randomly select n_trials indices to update
            trial_indices = rng.choice(pop_size, size=n_trials, replace=False)

            for idx in trial_indices:
                if time.time() > deadline:
                    break

                # Select three distinct random parents (different from idx)
                other_indices = [i for i in range(pop_size) if i != idx]
                p1, p2, p3 = rng.choice(other_indices, size=3, replace=False)

                # DE/rand/1: v = x[p1] + F_scale * (x[p2] - x[p3])
                v = X[p1] + F_scale * (X[p2] - X[p3])

                # Binary crossover with rate CR
                crossover_mask = rng.random(self.dim) < CR
                # Ensure at least one gene comes from the mutant vector
                if not np.any(crossover_mask):
                    j_rand = rng.randint(self.dim)
                    crossover_mask[j_rand] = True

                u = np.where(crossover_mask, v, X[idx])
                u = np.clip(u, lo, hi)

                # Evaluate trial vector
                fu = problem(u)
                used += 1
                all_F = np.vstack([all_F, fu.reshape(1, -1)])

                # Selection: keep the better of u and X[idx] based on dominance
                # u dominates X[idx]
                u_dominates = np.all(fu <= F[idx]) and np.any(fu < F[idx])
                # X[idx] dominates u
                x_dominates = np.all(F[idx] <= fu) and np.any(F[idx] < fu)

                if u_dominates or (not x_dominates and np.sum(fu) < np.sum(F[idx])):
                    X[idx] = u
                    F[idx] = fu

        return X
# EVOLVE-BLOCK-END
