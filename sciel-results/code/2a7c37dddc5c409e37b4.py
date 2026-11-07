"""Multi-objective optimizer for RE34 (3 objectives, 5 variables)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II-like multi-objective optimizer with adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        # Population size
        pop_size = 60
        # Archive for all non-dominated points found
        archive = []  # list of (x, f)
        archive_f = []  # list of f vectors

        # Initial population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Add initial non-dominated to archive
        for i in range(pop_size):
            self._add_to_archive(i, X, F)

        # Mutation rate (adaptive)
        mut_rate = 0.3
        crossover_rate = 0.8

        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # Find non-dominated front in current population
            fronts = self._non_dominated_fronts(F)
            # First front (non-dominated)
            front0_idx = [i for i in range(pop_size) if fronts[i] == 0]

            # Select parents from non-dominated set (or all if none)
            if len(front0_idx) == 0:
                parent_pool = list(range(pop_size))
            else:
                parent_pool = front0_idx

            # Generate children
            n_children = min(pop_size, self.budget - used)

            for _ in range(n_children):
                if time.time() > deadline:
                    break

                # Crossover: SBX-like blend
                if rng.random() < crossover_rate and len(parent_pool) >= 2:
                    p1_idx = parent_pool[rng.integers(len(parent_pool))]
                    p2_idx = parent_pool[rng.integers(len(parent_pool))]
                    if p1_idx == p2_idx:
                        p2_idx = parent_pool[(parent_pool.index(p2_idx) + 1) % len(parent_pool)]
                    # Blend
                    alpha = rng.random()
                    child = alpha * X[p1_idx] + (1 - alpha) * X[p2_idx]
                else:
                    p_idx = parent_pool[rng.integers(len(parent_pool))]
                    child = X[p_idx].copy()

                # Mutation
                if rng.random() < mut_rate:
                    # Gaussian mutation
                    scale = 0.15 * span
                    child += scale * rng.standard_normal(self.dim)
                else:
                    # Polynomial-like mutation on random subset
                    n_mut = max(1, self.dim // 2)
                    mut_idx = rng.choice(self.dim, size=n_mut, replace=False)
                    for j in mut_idx:
                        delta = rng.random() * 0.2 * span[j]
                        if rng.random() < 0.5:
                            child[j] += delta
                        else:
                            child[j] -= delta

                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1

                # Replace worst in population
                # Use a simple criterion: replace the point with largest sum of normalized objectives
                worst_idx = self._find_worst(F)
                X[worst_idx] = child
                F[worst_idx] = fc

                # Add to archive if non-dominated
                self._add_to_archive(used - 1, X, F)

            # Periodically reset population to maintain diversity
            if used % (2 * pop_size) < pop_size:
                # Replace some fraction with random points
                n_random = max(5, pop_size // 4)
                for i in range(n_random):
                    if used + i >= self.budget:
                        break
                    x_rand = lo + span * rng.random(self.dim)
                    f_rand = problem(x_rand)
                    used += 1
                    worst_idx = self._find_worst(F)
                    X[worst_idx] = x_rand
                    F[worst_idx] = f_rand
                    self._add_to_archive(used - 1, X, F)
                    if time.time() > deadline:
                        break

        # Final: ensure we haven't exceeded budget
        return X

    def _add_to_archive(self, idx, X, F):
        """Add point idx to archive if it is non-dominated by all archive points."""
        f_new = F[idx]
        x_new = X[idx]

        # Check if new point is dominated by any archive point
        for af in self.archive_f:
            if np.all(af <= f_new) and np.any(af < f_new):
                return  # dominated, don't add

        # Remove archive points dominated by new point
        new_archive_f = []
        for af in self.archive_f:
            if np.all(f_new <= af) and np.any(f_new < af):
                continue  # dominated by new point
            new_archive_f.append(af)

        self.archive_f = new_archive_f
        self.archive.append(x_new)

    def _non_dominated_fronts(self, F):
        """Assign front number to each point. Front 0 = non-dominated."""
        n = len(F)
        fronts = np.zeros(n, dtype=int)
        dominated = np.zeros(n, dtype=bool)

        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated[j] = True

        for i in range(n):
            if not dominated[i]:
                fronts[i] = 0
            else:
                fronts[i] = 1  # simplified: just mark as dominated

        return fronts

    def _find_worst(self, F):
        """Find the worst point to replace. Use a simple heuristic."""
        n = len(F)
        # Normalize objectives
        f_min = np.min(F, axis=0)
        f_max = np.max(F, axis=0)
        f_range = f_max - f_min
        f_range[f_range == 0] = 1.0
        F_norm = (F - f_min) / f_range
        scores = np.sum(F_norm, axis=1)
        return int(np.argmax(scores))
# EVOLVE-BLOCK-END
