"""Rewritten optimizer for DTLZ4 with deadline guard and smarter search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective optimizer for DTLZ4 with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(42)

        # Population size: use a good chunk of budget for initial population
        pop_size = min(80, max(20, self.budget // 3))

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        def time_check():
            return time.time() < deadline

        def fast_non_dominated_indices(F):
            """Return indices of non-dominated points using vectorized approach."""
            n = len(F)
            if n <= 1:
                return np.arange(n)
            # Point i is dominated if exists j such that F[j] <= F[i] and F[j] != F[i]
            # Vectorized: for each pair, check dominance
            # F[j] <= F[i] for all objectives AND F[j] < F[i] for at least one
            # This is O(n^2 * n_obj) but fast for n~80
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                # Check if any other point dominates i
                # F[j] <= F[i] for all j, and F[j] < F[i] for at least one j
                le = np.all(F <= F[i], axis=1)
                lt = np.any(F < F[i], axis=1)
                dom_mask = le & lt
                dom_mask[i] = False  # don't dominate self
                if np.any(dom_mask):
                    dominated[i] = True
            return np.where(~dominated)[0]

        def crowding_distance(F, indices):
            """Compute crowding distance for non-dominated front."""
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            n = len(indices)
            dist = np.zeros(n)
            F_sub = F[indices]
            for m in range(self.n_obj):
            # Sort by objective m
                order = np.argsort(F_sub[:, m])
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                range_val = F_sub[order[-1], m] - F_sub[order[0], m]
                if range_val > 1e-12:
                    for k in range(1, n - 1):
                        dist[order[k]] += (F_sub[order[k + 1], m] - F_sub[order[k - 1], m]) / range_val
            return dist

        # Main evolution loop
        while used < self.budget and time_check():
            # Find non-dominated front
            nd_indices = fast_non_dominated_indices(F)
            
            # Select parents from non-dominated set
            if len(nd_indices) > 0:
                # Use crowding distance to select diverse parents
                if len(nd_indices) > 2:
                    cd = crowding_distance(F, nd_indices)
                    # Select top half by crowding distance
                    top_k = max(2, len(nd_indices) // 2)
                    top_indices = nd_indices[np.argsort(cd)[-top_k:]]
                else:
                    top_indices = nd_indices
            else:
                top_indices = np.arange(len(F))

            # Create children using mutation from selected parents
            n_children = min(5, self.budget - used)
            children_X = []
            children_F = []
            
            for _ in range(n_children):
                parent_idx = top_indices[rng.integers(len(top_indices))]
                # Adaptive mutation: combine global and local steps
                # DTLZ4 has a bias toward extreme values due to x^20
                # Use a mix of small and larger steps
                scale = rng.choice([0.05, 0.1, 0.2, 0.4])
                noise = rng.standard_normal(self.dim) * scale * (hi - lo)
                child = X[parent_idx] + noise
                child = np.clip(child, lo, hi)
                
                # Also try a random restart with small probability
                if rng.random() < 0.1:
                    child = lo + (hi - lo) * rng.random(self.dim)
                
                children_X.append(child)
                children_F.append(problem(child))
                used += 1
                if used >= self.budget or not time_check():
                    break

            children_X = np.array(children_X)
            children_F = np.array(children_F)

            # Replace worst individuals
            if len(children_X) > 0:
                # Find dominated points to replace
                dominated_mask = np.ones(len(F), dtype=bool)
                for i in range(len(F)):
                    if not dominated_mask[i]:
                        continue
                    le = np.all(F <= F[i], axis=1)
                    lt = np.any(F < F[i], axis=1)
                    dom_mask = le & lt
                    dom_mask[i] = False
                    if not np.any(dom_mask):
                        dominated_mask[i] = False

                # Prefer replacing dominated points
                if np.any(dominated_mask):
                    replace_pool = np.where(dominated_mask)[0]
                else:
                    replace_pool = np.arange(len(F))

                for k in range(len(children_X)):
                    # Replace a random point from the pool
                    idx = replace_pool[rng.integers(len(replace_pool))]
                    F[idx] = children_F[k]
                    X[idx] = children_X[k]
                    # Remove from pool if possible
                    replace_pool = replace_pool[replace_pool != idx]
                    if len(replace_pool) == 0:
                        replace_pool = np.arange(len(F))

        return X
# EVOLVE-BLOCK-END
