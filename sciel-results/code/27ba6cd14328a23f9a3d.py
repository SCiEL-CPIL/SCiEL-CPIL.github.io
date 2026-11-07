"""Multi-objective optimizer using Pareto Local Search with adaptive step size and diversity-based parent selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer: Local search with Pareto front maintenance."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(0)

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume (used for potential scoring, but not directly for search here)
        # ref = np.array([1705.0, 12.0, 0.3])

        # Initialize with Latin Hypercube Sampling for good spread
        n_init = min(30, max(10, budget // 6))
        # LHS in [0, 1]^dim
        X_init = np.zeros((n_init, dim))
        for j in range(dim):
            perm = rng.permutation(n_init)
            X_init[:, j] = (perm + 0.5) / n_init
        
        # Map to actual bounds
        X_init = lo + (hi - lo) * X_init
        F_init = np.array([problem(x) for x in X_init])
        used = n_init

        # Archive: list of (x, f) tuples that are non-dominated
        archive_X = [X_init[i].copy() for i in range(n_init)]
        archive_F = [F_init[i].copy() for i in range(n_init)]

        def dominates(a, b):
            """Check if a dominates b."""
            return np.all(a <= b) and np.any(a < b)

        def is_non_dominated(point, front):
            """Check if point is non-dominated with respect to front."""
            for f in front:
                if dominates(f, point):
                    return False
            return True

        def remove_dominated(point, front):
            """Remove points from front that are dominated by point."""
            return [f for f in front if not dominates(point, f)]

        def update_archive(x_new, f_new, archive_X, archive_F):
            """Update the Pareto front. Returns new lists."""
            if not is_non_dominated(f_new, archive_F):
                return archive_X, archive_F
            
            # Remove dominated points
            new_F = remove_dominated(f_new, archive_F)
            
            # We need to keep X and F aligned. 
            # Since we only have F in the filter, we need to be careful.
            # Let's rebuild from scratch for simplicity and correctness, 
            # as the archive size is small (< 200).
            
            # Combine
            all_X = archive_X + [x_new.copy()]
            all_F = archive_F + [f_new.copy()]
            
            # Filter non-dominated
            final_X = []
            final_F = []
            for i in range(len(all_F)):
                if is_non_dominated(all_F[i], all_F):
                    final_X.append(all_X[i])
                    final_F.append(all_F[i])
            
            return final_X, final_F

        def select_parent(archive_F):
            """Select a parent based on largest neighborhood (diversity)."""
            if len(archive_F) == 1:
                return 0
            
            # Calculate nearest neighbor distance for each point in objective space
            n = len(archive_F)
            dists = np.zeros(n)
            for i in range(n):
                min_dist = np.inf
                for j in range(n):
                    if i == j: continue
                    d = np.linalg.norm(archive_F[i] - archive_F[j])
                    if d < min_dist:
                        min_dist = d
                dists[i] = min_dist
            
            # Select point with largest distance to nearest neighbor
            # Add some randomness to avoid getting stuck in a loop
            # Use a tournament or softmax
            # Simple: pick argmax, but with 20% chance pick random
            if rng.random() < 0.2:
                return rng.randint(n)
            return np.argmax(dists)

        # Adaptive step size parameters
        sigma_start = 0.5 * np.max(hi - lo)
        sigma_end = 0.05 * np.min(hi - lo)
        
        # Main optimization loop
        while used < budget:
            if time.time() > deadline:
                break

            # Select parent
            parent_idx = select_parent(archive_F)
            parent = archive_X[parent_idx]
            
            # Calculate adaptive step size
            progress = used / budget
            sigma = sigma_start * (1 - progress) + sigma_end * progress
            
            # Generate child
            noise = rng.standard_normal(dim) * sigma
            child = parent + noise
            child = np.clip(child, lo, hi)
            
            # Evaluate
            f_child = problem(child)
            used += 1
            
            # Update archive
            archive_X, archive_F = update_archive(child, f_child, archive_X, archive_F)

        return X_init
# EVOLVE-BLOCK-END
