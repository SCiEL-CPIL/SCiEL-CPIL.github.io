"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer with LHS init, diversity-aware replacement, and deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # Latin Hypercube initialization for better spread
        pop_size = min(40, self.budget // 3)
        X = np.zeros((pop_size, dim))
        for j in range(dim):
            perm = rng.permutation(pop_size)
            X[:, j] = (perm + rng.random(pop_size)) / pop_size
        X = lo + (hi - lo) * X

        F = np.array([problem(x) for x in X])
        used = pop_size

        # Diversity-aware replacement: replace the point whose removal would
        # cause the largest increase in mean pairwise distance (or equivalently,
        # replace the point closest to the population mean in objective space)
        while used < self.budget:
            if time.time() > deadline - 5.0:
                break

            # Find non-dominated points
            dominated = np.zeros(pop_size, dtype=bool)
            for i in range(pop_size):
                for j in range(pop_size):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break

            if np.any(~dominated):
                parents_idx = np.where(~dominated)[0]
            else:
                parents_idx = np.arange(pop_size)

            parent_idx = parents_idx[rng.integers(len(parents_idx))]

            # Mutation with adaptive step size
            step = 0.1 * (hi - lo)
            child = X[parent_idx] + step * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Diversity-based replacement: replace the point closest to the
            # population mean in objective space (removing it changes the front least)
            F_mean = np.mean(F, axis=0)
            distances = np.sum((F - F_mean) ** 2, axis=1)
            worst = int(np.argmin(distances))
            X[worst], F[worst] = child, fc

        return X
# EVOLVE-BLOCK-END
