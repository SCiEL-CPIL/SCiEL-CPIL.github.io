"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Fast multi-objective optimizer: non-dominated parent mutation with reference-aware replacement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 285.0  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        ref = np.array([1.1, 1.3, 1.1])

        dim = self.dim
        n_obj = self.n_obj

        # Initial population: balanced between exploration and budget
        pop_size = min(30, self.budget // 5)
        X = lo + span * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def is_nondominated(i, F):
            """Check if point i is non-dominated in F."""
            fi = F[i]
            for j in range(len(F)):
                if i == j:
                    continue
                fj = F[j]
                if np.all(fj <= fi) and np.any(fj < fi):
                    return False
            return True

        def get_nondominated_indices(F):
            """Get indices of non-dominated points."""
            return [i for i in range(len(F)) if is_nondominated(i, F)]

        def get_replacement_index(F, new_point):
            """
            Decide which point to replace.
            Priority 1: Replace points worse than reference (useless for HV).
            Priority 2: Replace dominated points.
            Priority 3: Replace point with highest normalized sum (least useful).
            """
            n = len(F)
            new_better = np.all(new_point < ref)

            # Find points NOT strictly better than ref in all objectives
            bad_indices = [i for i in range(n) if not np.all(F[i] < ref)]

            if bad_indices:
                # Among bad points, replace the one with highest sum (worst)
                sums = np.sum(F[bad_indices], axis=1)
                worst_local = int(np.argmax(sums))
                return bad_indices[worst_local]
            else:
                # All points are better than ref
                # Check if new point dominates any existing point
                dominated_by_new = []
                for i in range(n):
                    if np.all(new_point <= F[i]) and np.any(new_point < F[i]):
                        dominated_by_new.append(i)

                if dominated_by_new:
                    # Replace one of the dominated points (the worst among them)
                    sums = np.sum(F[dominated_by_new], axis=1)
                    worst_local = int(np.argmax(sums))
                    return dominated_by_new[worst_local]
                else:
                    # Replace point with highest normalized sum
                    # Normalize by span for fairness
                    normalized = F / span
                    sums = np.sum(normalized, axis=1)
                    return int(np.argmax(sums))

        # Adaptive mutation: start larger, shrink over time
        total_budget = self.budget
        while used < self.budget:
            if time.time() > deadline:
                break

            # Progress: 0 at start, 1 at end
            progress = used / total_budget

            # Find non-dominated parents
            nd_idx = get_nondominated_indices(F)
            if len(nd_idx) == 0:
                nd_idx = list(range(len(F)))

            parents = X[nd_idx]

            # Select parent (random from non-dominated)
            p_idx = rng.integers(len(parents))
            parent = parents[p_idx].copy()

            # Adaptive mutation: larger early, smaller late
            # Base sigma: 0.15 * span, shrinking to 0.02 * span
            sigma_scale = 0.15 - 0.13 * progress
            sigma = sigma_scale * span
            child = parent + sigma * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)

            # Evaluate
            fc = problem(child)
            used += 1

            # Replace worst point
            worst_idx = get_replacement_index(F, fc)
            X[worst_idx], F[worst_idx] = child, fc

        return X
# EVOLVE-BLOCK-END
