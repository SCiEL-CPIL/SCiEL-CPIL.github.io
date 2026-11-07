"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 300.0  # Safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(40, budget // 3)
        if pop_size < 5:
            pop_size = 5

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Reference point for hypervolume (only for local selection, not returned)
        ref = np.array([3.0, 5.0, 7.0])

        def fast_non_dominated(F):
            """Return indices of non-dominated points."""
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated[j] = True
            return ~dominated

        def crowding_distance(F, nd_idx):
            """Compute crowding distance for non-dominated points."""
            n = len(nd_idx)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            F_nd = F[nd_idx]
            for m in range(F_nd.shape[1]):
                min_v = F_nd[:, m].min()
                max_v = F_nd[:, m].max()
                if max_v - min_v < 1e-12:
                    continue
                order = np.argsort(F_nd[:, m])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for k in range(1, n - 1):
                    cd[order[k]] += (F_nd[order[k + 1], m] - F_nd[order[k - 1], m]) / (max_v - min_v)
            return cd

        def select_parents(F, X):
            """Select parents using non-dominated sorting + crowding."""
            nd = fast_non_dominated(F)
            if np.sum(nd) >= 2:
                nd_idx = np.where(nd)[0]
                cd = crowding_distance(F, nd_idx)
                # Weighted random selection from non-dominated
                weights = cd + 1e-6
                weights /= weights.sum()
                return X[nd_idx], rng.choice(len(nd_idx), p=weights)
            else:
                return X, rng.integers(len(X))

        while used < budget:
            # Check deadline with safety margin
            if time.time() > deadline - 5.0:
                break

            # Select parents
            parents_X, pidx = select_parents(F, X)
            parent = parents_X[pidx]

            # Mutate: adaptive step size based on progress
            # Use a mix of large and small steps
            if rng.random() < 0.3:
                step_scale = 0.3
            else:
                step_scale = 0.05

            mutation = step_scale * (hi - lo) * rng.standard_normal(dim)
            child = parent + mutation
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace worst point in population
            # Use a simple metric: sum of normalized objectives
            # Points outside reference point get penalized
            F_clipped = np.minimum(F, ref)
            normalized = F_clipped / ref
            scores = normalized.sum(axis=1)
            worst = int(np.argmax(scores))

            X[worst] = child
            F[worst] = fc

        return X
# EVOLVE-BLOCK-END
