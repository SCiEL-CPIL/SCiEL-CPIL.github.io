"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective search with adaptive mutation and time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # 300s limit, 20s safety margin
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Initial population
        pop_size = min(30, self.budget // 4)
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def is_dominated(f, F_all):
            """Check if f is dominated by any point in F_all."""
            return np.any(np.all(F_all <= f, axis=1) & np.any(F_all < f, axis=1))

        def non_dominated_indices(F_all):
            """Return indices of non-dominated points."""
            n = len(F_all)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j or not nd[j]:
                        continue
                    if np.all(F_all[j] <= F_all[i]) and np.any(F_all[j] < F_all[i]):
                        nd[i] = False
            return np.where(nd)[0]

        # Adaptive mutation step size
        base_step = 0.3
        min_step = 0.01
        step = base_step

        while used < self.budget:
            # Time guard
            if time.time() > deadline:
                break

            # Find non-dominated parents
            nd_idx = non_dominated_indices(F)
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(F))
            parents = X[nd_idx]
            parent_F = F[nd_idx]

            # Select parent with tournament selection on hypervolume contribution
            # Use simple random selection from non-dominated set
            p_idx = rng.integers(len(parents))
            parent = parents[p_idx]

            # Adaptive mutation: Gaussian with current step size
            mutation = step * span * rng.standard_normal(self.dim)
            child = parent + mutation
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replacement: remove the point with worst contribution
            # Use a simple metric: sum of objectives (lower is better)
            # But better: use a crowding-distance-like measure
            # For simplicity, use sum of normalized objectives
            sums = np.sum(F, axis=1)
            
            # Check if child is non-dominated
            child_dominated = False
            for i in range(len(F)):
                if np.all(F[i] <= fc) and np.any(F[i] < fc):
                    child_dominated = True
                    break

            if not child_dominated:
                # Child is non-dominated: replace the worst dominated point if exists
                dominated_mask = np.array([is_dominated(F[i], F) for i in range(len(F))])
                if np.any(dominated_mask):
                    # Replace the worst dominated point
                    worst_dominated = np.where(dominated_mask)[0]
                    worst_idx = worst_dominated[np.argmax(sums[worst_dominated])]
                else:
                    # All non-dominated, replace the one with highest sum
                    worst_idx = int(np.argmax(sums))
            else:
                # Child is dominated: only replace if it's better than worst
                worst_idx = int(np.argmax(sums))
                if np.sum(fc) < sums[worst_idx]:
                    X[worst_idx], F[worst_idx] = child, fc
                    continue
                else:
                    # Don't replace, but still count the evaluation
                    # Reduce step size slightly
                    step = max(min_step, step * 0.95)
                    continue

            X[worst_idx], F[worst_idx] = child, fc

            # Adapt step size: if child improved the worst, increase; else decrease
            if np.sum(fc) < np.sum(F[worst_idx]):
                step = min(base_step, step * 1.05)
            else:
                step = max(min_step, step * 0.98)

        return X
# EVOLVE-BLOCK-END
