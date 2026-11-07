"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG4 using a population-based evolutionary strategy
    with non-dominated sorting and crowding distance."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        deadline = time.time() + 270  # Safety margin before 300s limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Hypervolume contribution function for 3 objectives
        def hv_contribution(f, archive):
            """Compute the hypervolume contribution of point f to the archive."""
            if len(archive) == 0:
                return (3.0 - f[0]) * (5.0 - f[1]) * (7.0 - f[2]) if all(f < [3.0, 5.0, 7.0]) else 0.0
            # Simple approach: check if f is dominated by any point in archive
            for a in archive:
                if np.all(a <= f) and np.any(a < f):
                    return 0.0
            # Approximate contribution: product of differences to reference point
            # minus overlap with existing archive points
            # For simplicity, use a heuristic estimate
            vol = 1.0
            for i in range(n_obj):
                ref = [3.0, 5.0, 7.0][i]
                min_val = f[i]
                for a in archive:
                    if a[i] < f[i]:
                        min_val = max(min_val, a[i])
                vol *= (ref - min_val)
            return max(0.0, vol)

        # Non-dominated sort
        def non_dominated_sort(F):
            """Return list of fronts (indices) sorted by dominance."""
            n = len(F)
            S = [[] for _ in range(n)]  # Dominated sets
            n_dominates = [0] * n  # Number of points dominated
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dominates[i] += 1
            
            current_front = [i for i in range(n) if n_dominates[i] == 0]
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            next_front.append(j)
                current_front = next_front
            return fronts

        # Crowding distance
        def crowding_distance(F, indices):
            """Compute crowding distance for points at given indices."""
            if len(indices) <= 2:
                return [float('inf')] * len(indices)
            
            n = len(indices)
            dist = [0.0] * n
            F_subset = F[indices]
            
            for m in range(n_obj):
                order = np.argsort(F_subset[:, m])
                dist[order[0]] = float('inf')
                dist[order[-1]] = float('inf')
                range_val = F_subset[order[-1], m] - F_subset[order[0], m]
                if range_val == 0:
                    continue
                for k in range(1, n - 1):
                    dist[order[k]] += (F_subset[order[k+1], m] - F_subset[order[k-1], m]) / range_val
            
            return dist

        # Initialize population
        rng = np.random.default_rng()
        pop_size = min(50, budget // 4)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Keep track of best points found (archive)
        archive = []
        archive_F = []

        def update_archive(x, f):
            nonlocal archive, archive_F
            # Check if f is dominated by any point in archive
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return
            # Remove points dominated by f
            new_archive = []
            new_archive_F = []
            for i, af in enumerate(archive_F):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive.append(archive[i])
                    new_archive_F.append(af)
            new_archive.append(x)
            new_archive_F.append(f)
            archive = new_archive
            archive_F = new_archive_F

        for x, f in zip(X, F):
            update_archive(x, f)

        # Main evolution loop
        while used < budget:
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts = non_dominated_sort(F)
            
            # Select parents from first front
            if fronts:
                first_front = fronts[0]
                if len(first_front) > 1:
                    cd = crowding_distance(F, first_front)
                    # Tournament selection based on crowding distance
                    i1 = first_front[rng.integers(len(first_front))]
                    i2 = first_front[rng.integers(len(first_front))]
                    parent_idx = i1 if cd[first_front.index(i1)] > cd[first_front.index(i2)] else i2
                else:
                    parent_idx = first_front[0]
            else:
                parent_idx = 0

            # Create child via mutation
            child = X[parent_idx].copy()
            # Adaptive mutation rate
            rate = 0.3 * (1.0 - used / budget) + 0.05
            mask = rng.random(dim) < 0.5
            child[mask] += rate * (hi[mask] - lo[mask]) * rng.standard_normal(np.sum(mask))
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace worst point in population
            # Use hypervolume-based or simple replacement
            # Find the point with worst hypervolume contribution
            worst_idx = 0
            worst_score = float('inf')
            for i in range(len(F)):
                # Simple score: sum of objectives (lower is better)
                score = np.sum(F[i])
                if score > worst_score:
                    worst_score = score
                    worst_idx = i
            
            X[worst_idx] = child
            F[worst_idx] = fc
            update_archive(child, fc)

        # Return the archive (best non-dominated points)
        if archive:
            return np.array(archive)
        else:
            return X

# EVOLVE-BLOCK-END
