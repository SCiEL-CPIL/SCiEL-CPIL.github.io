"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective search with adaptive mutation and hypervolume-aware replacement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        pop_size = int(min(30, self.budget // 6))
        if pop_size < 5:
            pop_size = 5

        X = lo + range_ * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def get_non_dominated_indices(F_mat):
            n = len(F_mat)
            non_dominated = np.ones(n, dtype=bool)
            for i in range(n):
                if not non_dominated[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_mat[j] <= F_mat[i]) and np.any(F_mat[j] < F_mat[i]):
                        non_dominated[i] = False
                        break
            return np.where(non_dominated)[0]

        while used < self.budget:
            if time.time() > deadline:
                break

            nd_idx = get_non_dominated_indices(F)
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(F))
            parents = X[nd_idx]

            parent_idx = rng.integers(len(parents))
            parent = parents[parent_idx]

            progress = used / self.budget
            base_step = 0.3 * (1 - progress) + 0.05
            if rng.random() < 0.1:
                step_size = base_step * 3
            else:
                step_size = base_step

            child = parent + step_size * range_ * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            dominated_mask = np.array([
                np.all(F[i] >= fc) and np.any(F[i] > fc) for i in range(len(F))
            ])

            if np.any(dominated_mask):
                dominated_indices = np.where(dominated_mask)[0]
                replace_idx = dominated_indices[rng.integers(len(dominated_indices))]
            else:
                sums = np.sum(F, axis=1)
                replace_idx = int(np.argmax(sums))
                if np.sum(fc) >= sums[replace_idx]:
                    continue

            X[replace_idx] = child
            F[replace_idx] = fc

        return X
# EVOLVE-BLOCK-END
