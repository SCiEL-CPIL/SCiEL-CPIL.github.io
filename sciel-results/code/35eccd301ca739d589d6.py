"""Multi-objective optimizer using differential evolution with hypervolume-guided selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer: DE-based search with Pareto archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(0)

        dim = self.dim
        budget = self.budget

        # Reference point for hypervolume
        ref = np.array([1705.0, 12.0, 0.3])

        # Initialize with stratified sampling for good spread
        n_init = min(30, max(10, budget // 7))
        # Use Latin Hypercube-like initialization
        n_strata = max(2, int(round(n_init ** (1.0 / dim))))
        grid = np.array(np.meshgrid(*[np.linspace(0.5/n_strata, 1-0.5/n_strata, n_strata) for _ in range(dim)]), indexing='ij')
        grid = grid.reshape(dim, -1).T
        # Shuffle and take first n_init
        rng.shuffle(grid)
        grid = grid[:n_init]
        # Add small jitter
        grid = grid + rng.random((n_init, dim)) * 0.05
        grid = np.clip(grid, 0, 1)

        X_init = lo + (hi - lo) * grid
        F_init = np.array([problem(x) for x in X_init])
        used = n_init

        # Maintain a list of all evaluated points
        all_X = [X_init[i].copy() for i in range(n_init)]
        all_F = [F_init[i].copy() for i in range(n_init)]

        def dominates(a, b):
            """Check if a dominates b (a <= b in all, a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def get_pareto_front(all_F):
            """Get indices of non-dominated points."""
            n = len(all_F)
            non_dom = np.ones(n, dtype=bool)
            for i in range(n):
                if not non_dom[i]:
                    continue
                for j in range(n):
                    if i == j or not non_dom[j]:
                        continue
                    if dominates(all_F[j], all_F[i]):
                        non_dom[i] = False
                        break
            return np.where(non_dom)[0]

        def update_archive(X_new, F_new):
            """Add new point to archive if it's non-dominated."""
            # Check if new point is dominated by any existing point
            for f in all_F:
                if dominates(f, F_new):
                    return
            # Remove points dominated by new point
            new_all_X = []
            new_all_F = []
            for i in range(len(all_X)):
                if not dominates(F_new, all_F[i]):
                    new_all_X.append(all_X[i])
                    new_all_F.append(all_F[i])
            new_all_X.append(X_new.copy())
            new_all_F.append(F_new.copy())
            all_X.clear()
            all_F.clear()
            all_X.extend(new_all_X)
            all_F.extend(new_all_F)

        # DE parameters
        CR = 0.9
        F_mut = 0.8
        max_archive_size = 50

        # Main optimization loop
        while used < budget:
            if time.time() > deadline:
                break

            # Get current Pareto front
            front_idx = get_pareto_front(all_F)
            n_front = len(front_idx)

            if n_front < 3:
                # Not enough diversity, do random exploration
                trial = lo + (hi - lo) * rng.random(dim)
            else:
                # Select parents from front
                if n_front > 10:
                    # Sample 3 from front
                    parents = rng.choice(front_idx, size=3, replace=False)
                else:
                    parents = front_idx
                    while len(parents) < 3:
                        parents = np.append(parents, parents[0])

                a, b, c = parents[0], parents[1], parents[2]
                v = all_X[a] + F_mut * (all_X[b] - all_X[c])
                # Binomial crossover with random parent
                j_rand = rng.randint(0, dim)
                trial = v.copy()
                for j in range(dim):
                    if rng.random() <= CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = all_X[a][j]
                trial = np.clip(trial, lo, hi)

            # Evaluate
            f_trial = problem(trial)
            used += 1

            # Update archive
            update_archive(trial, f_trial)

            # If archive grows too large, prune it
            if len(all_F) > max_archive_size:
                front_idx = get_pareto_front(all_F)
                if len(front_idx) > max_archive_size:
                    # Keep only max_archive_size points from front
                    # Use crowding distance approximation: keep points that are spread out
                    # For simplicity, just keep the first max_archive_size
                    keep = front_idx[:max_archive_size]
                    new_X = [all_X[i] for i in keep]
                    new_F = [all_F[i] for i in keep]
                    all_X.clear()
                    all_F.clear()
                    all_X.extend(new_X)
                    all_F.extend(new_F)

        return all_X
# EVOLVE-BLOCK-END
