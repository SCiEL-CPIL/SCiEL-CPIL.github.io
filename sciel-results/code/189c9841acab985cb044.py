"""Rewritten optimizer for ZDT4 with deadline guard and efficient MOEA."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT4 using a simple non-dominated sorting approach."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        deadline = time.time() + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Initial population
        pop_size = min(40, self.budget // 5)
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominates(a, b):
            """Check if a dominates b (both minimize)."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F):
            """Return indices of non-dominated points."""
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(i + 1, n):
                    if dominated[j]:
                        continue
                    if dominates(F[i], F[j]):
                        dominated[j] = True
                    elif dominates(F[j], F[i]):
                        dominated[i] = True
                        break
            return np.where(~dominated)[0]

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Get non-dominated set
            nd_idx = non_dominated_sort(F)
            if len(nd_idx) == 0:
                nd_idx = np.array([0])

            # Select parent from non-dominated set
            parent_idx = nd_idx[rng.integers(len(nd_idx))]
            parent = X[parent_idx]

            # Mutation: Gaussian perturbation scaled by span
            sigma = 0.05
            child = parent + sigma * span * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)

            # Evaluate child
            fc = problem(child)
            used += 1

            # Replace worst point (highest sum of objectives)
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = child
            F[worst_idx] = fc

            # Periodically re-seed with random points to maintain diversity
            if used % 50 == 0 and used < self.budget - 10:
                n_new = min(5, self.budget - used)
                if n_new > 0:
                    for _ in range(n_new):
                        if time.time() > deadline:
                            break
                        new_x = lo + span * rng.random(self.dim)
                        new_f = problem(new_x)
                        used += 1
                        worst_idx = int(np.argmax(np.sum(F, axis=1)))
                        X[worst_idx] = new_x
                        F[worst_idx] = new_f

        return X
# EVOLVE-BLOCK-END
