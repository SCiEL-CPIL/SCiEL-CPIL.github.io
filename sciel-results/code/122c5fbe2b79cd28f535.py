"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer using NSGA-II with DE-inspired mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng_span = hi - lo

        # Population size: use ~50% of budget for initial pop, rest for evolution
        pop_size = min(60, max(10, self.budget // 4))
        remaining = self.budget - pop_size

        # Initialize population
        X = lo + rng_span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        evals = pop_size

        def non_dominated_sort(F):
            """Return non-dominated indices."""
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            return np.where(~dominated)[0]

        def crowding_distance(F, nd_idx):
            """Compute crowding distance for non-dominated points."""
            if len(nd_idx) <= 2:
                return np.full(len(nd_idx), np.inf)
            cd = np.zeros(len(nd_idx))
            for m in range(self.n_obj):
                sorted_idx = np.argsort(F[nd_idx, m])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                for k in range(1, len(sorted_idx) - 1):
                    f_range = F[nd_idx[sorted_idx[-1]], m] - F[nd_idx[sorted_idx[0]], m]
                    if f_range > 0:
                        cd[sorted_idx[k]] += (
                            F[nd_idx[sorted_idx[k + 1]], m] - F[nd_idx[sorted_idx[k - 1]], m]
                        ) / f_range
            return cd

        while evals < self.budget:
            # Get non-dominated set
            nd_idx = non_dominated_sort(F)
            
            # Select parents from non-dominated set using crowding distance
            if len(nd_idx) > 1:
                cd = crowding_distance(F, nd_idx)
                # Tournament selection: pick 2, choose one with higher crowding distance
                p1_idx = nd_idx[rng.integers(len(nd_idx))]
                p2_idx = nd_idx[rng.integers(len(nd_idx))]
                # DE mutation: x_child = x_p1 + F * (x_p1 - x_p2)
                F_param = 0.8 * rng.random() + 0.2
                child = X[p1_idx] + F_param * (X[p1_idx] - X[p2_idx])
            else:
                # Fallback: simple mutation
                p_idx = nd_idx[0] if len(nd_idx) > 0 else rng.integers(pop_size)
                child = X[p_idx] + 0.1 * rng_span * rng.standard_normal(self.dim)
            
            child = np.clip(child, lo, hi)
            fc = problem(child)
            evals += 1

            # Replace worst point in population
            # Use a simple strategy: replace the point that is dominated and has worst sum
            dominated_mask = np.zeros(pop_size, dtype=bool)
            for i in range(pop_size):
                for j in range(pop_size):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated_mask[i] = True
                        break
            
            if np.any(dominated_mask):
                candidates = np.where(dominated_mask)[0]
                worst = candidates[np.argmax(np.sum(F[candidates], axis=1))]
            else:
                worst = np.argmax(np.sum(F, axis=1))
            
            X[worst] = child
            F[worst] = fc

        return X
# EVOLVE-BLOCK-END
