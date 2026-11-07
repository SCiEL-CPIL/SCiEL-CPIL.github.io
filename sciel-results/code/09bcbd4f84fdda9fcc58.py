"""Multi-objective optimizer on RE21 using archive-based DE with HV-guided selection."""
import numpy as np
import time


class Optimizer:
    """Differential Evolution with archive-based Pareto front maintenance and HV-guided selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        ref = np.array([3000.0, 0.06])
        rng = np.random.default_rng(42)

        # Archive to store non-dominated points
        archive_X = []
        archive_F = []
        used = 0

        def add_to_archive(x, f):
            """Add point to archive if non-dominated, remove dominated points."""
            nonlocal used
            # Check if new point dominates any archive point
            i = 0
            while i < len(archive_F):
                if np.all(f <= archive_F[i]) and np.any(f < archive_F[i]):
                    archive_X.pop(i)
                    archive_F.pop(i)
                else:
                    i += 1
            # Check if any archive point dominates new point
            for i in range(len(archive_F)):
                if np.all(archive_F[i] <= f) and np.any(archive_F[i] < f):
                    return
            # Add new point
            archive_X.append(x.copy())
            archive_F.append(f.copy())

        def hypervolume_2d(points, ref_pt):
            """Calculate hypervolume for 2D points."""
            if len(points) == 0:
                return 0.0
            mask = np.all(points < ref_pt, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            pts = pts[np.argsort(pts[:, 0])]
            hv = 0.0
            prev_y = ref_pt[1]
            for i in range(len(pts)):
                x1, y1 = pts[i]
                if y1 < prev_y:
                    hv += (ref_pt[0] - x1) * (prev_y - y1)
                    prev_y = y1
            return hv

        def hv_contribution(f_new, archive_F_arr, ref_pt):
            """Calculate HV contribution of a new point given current archive."""
            hv_before = hypervolume_2d(archive_F_arr, ref_pt)
            # Add new point and compute HV
            new_points = np.vstack([archive_F_arr, f_new.reshape(1, -1)])
            hv_after = hypervolume_2d(new_points, ref_pt)
            return hv_after - hv_before

        def non_dominated_mask(F):
            """Return boolean mask of non-dominated points."""
            n = len(F)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                for j in range(n):
                    if i == j or not mask[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        mask[i] = False
                        break
            return mask

        # Phase 1: Reference-point-aligned grid initialization
        # Since ref=[3000, 0.06], obj1 range is much larger than obj2.
        # We want to explore low obj1 values heavily.
        n_init = min(30, self.budget // 6)
        if n_init < 10:
            n_init = 10

        # Create a grid that emphasizes low obj1 region
        # Use stratified sampling in decision space
        X_init = np.empty((n_init, self.dim))
        for i in range(n_init):
            # Mix of uniform and boundary-focused sampling
            if i % 3 == 0:
                # Near lower bounds
                X_init[i] = lo + (hi - lo) * rng.random(self.dim) * 0.3
            elif i % 3 == 1:
                # Mid-range
                X_init[i] = lo + (hi - lo) * (0.3 + 0.4 * rng.random(self.dim))
            else:
                # Full range
                X_init[i] = lo + (hi - lo) * rng.random(self.dim)

        for i in range(n_init):
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            f = problem(X_init[i])
            used += 1
            add_to_archive(X_init[i], f)

        # Phase 2: Adaptive DE loop with HV-guided parent selection
        F_mut = 0.8
        CR = 0.9
        pop_size = min(20, max(10, len(archive_X) if len(archive_X) > 0 else 10))

        while used < self.budget:
            if time.time() > deadline:
                break

            # Get current archive as population
            if len(archive_X) == 0:
                break

            n_arch = len(archive_X)
            if n_arch < 3:
                # Not enough diversity, do random restarts
                n_random = min(5, self.budget - used)
                for _ in range(n_random):
                    if used >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    x = lo + (hi - lo) * rng.random(self.dim)
                    f = problem(x)
                    used += 1
                    add_to_archive(x, f)
                continue

            # Generate offspring
            n_offspring = min(pop_size, self.budget - used)

            for _ in range(n_offspring):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break

                # Select parents from archive
                # Use HV contribution to guide selection: prefer points with high contribution
                if n_arch >= 3:
                    # Compute approximate HV contributions for selection
                    # Use a simplified approach: select based on objective values
                    # Points with lower obj1 tend to have higher HV contribution
                    # Use tournament selection based on a combined score
                    n_tournament = min(3, n_arch)
                    candidates = rng.choice(n_arch, size=n_tournament, replace=False)
                    # Score: lower obj1 is better (since it dominates HV)
                    scores = archive_F[candidates, 0] + 0.01 * archive_F[candidates, 1]
                    best_idx = candidates[np.argmin(scores)]

                    # Select 3 distinct parents for DE
                    remaining = [i for i in range(n_arch) if i != best_idx]
                    if len(remaining) >= 2:
                        other_indices = rng.choice(remaining, size=2, replace=False)
                        parents = [best_idx, other_indices[0], other_indices[1]]
                    else:
                        parents = rng.choice(n_arch, size=3, replace=False)
                else:
                    parents = rng.choice(n_arch, size=3, replace=True)

                p1, p2, p3 = parents[0], parents[1], parents[2]
                x1, x2, x3 = archive_X[p1], archive_X[p2], archive_X[p3]

                # Mutation
                mutant_x = x1 + F_mut * (x2 - x3)

                # Crossover
                trial = np.empty(self.dim)
                jrand = rng.integers(self.dim)
                for j in range(self.dim):
                    if rng.random() < CR or j == jrand:
                        trial[j] = mutant_x[j]
                    else:
                        trial[j] = x1[j]

                # Clip to bounds
                trial = np.clip(trial, lo, hi)

                # Evaluate
                f_trial = problem(trial)
                used += 1

                # Check if trial is non-dominated by archive
                dominated = False
                for i in range(len(archive_F)):
                    if np.all(archive_F[i] <= f_trial) and np.any(archive_F[i] < f_trial):
                        dominated = True
                        break

                if not dominated:
                    add_to_archive(trial, f_trial)
                else:
                    # Even if dominated, check if it improves HV (might be useful for exploration)
                    # Simple heuristic: accept if it's in a less crowded region
                    pass

            # Adaptive mutation rate: shrink over time to exploit
            progress = used / self.budget
            F_mut = 0.8 * (1.0 - progress) + 0.3 * progress
            CR = 0.9 * (1.0 - progress) + 0.7 * progress

            # Periodic random exploration to maintain diversity
            if used % 20 == 0 and used < self.budget - 10:
                n_explore = min(3, self.budget - used)
                for _ in range(n_explore):
                    if used >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    x = lo + (hi - lo) * rng.random(self.dim)
                    f = problem(x)
                    used += 1
                    add_to_archive(x, f)

        return archive_X[0] if archive_X else lo
