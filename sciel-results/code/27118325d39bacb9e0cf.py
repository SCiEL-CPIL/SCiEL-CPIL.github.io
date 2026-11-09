"""Multi-objective optimizer for RE21 using a direct grid + local refinement approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Direct multi-objective optimizer for RE21 (4 vars, 2 objectives)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        ref_point = np.array([3000.0, 0.06])

        # We'll track all evaluated points
        all_X = []
        all_F = []
        used = 0

        def eval_point(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = problem(x)
            f = np.array(f, dtype=np.float64)
            all_X.append(x.copy())
            all_F.append(f.copy())
            used += 1
            return f

        # Check if any point is already good enough (better than ref in all objectives)
        def is_valid(f):
            return np.all(f < ref_point)

        # Phase 1: Stratified sampling to explore the space
        # For 4 dimensions, use a structured grid with 3 points per dim -> 3^4 = 81 points
        # But budget is 200, so we can afford more. Let's do 4^4 = 256? No, budget is 200.
        # Let's do 3^4 = 81 points for initial exploration.
        
        # Actually, a smarter approach: use Latin Hypercube Sampling or a grid.
        # Given the budget of 200, let's do a two-phase approach:
        # Phase 1: 100 points via stratified/grid sampling
        # Phase 2: 100 points via local search around best points

        # Phase 1: Stratified sampling for broad coverage
        # Use 60 points with stratification in each dimension
        n_strat = 60
        for i in range(n_strat):
            if used >= self.budget or time.time() > deadline:
                break
            # Each dimension gets a different stratum index, shuffled
            x = np.zeros(self.dim)
            for d in range(self.dim):
                # Assign stratum based on a shuffled permutation
                stratum = (i * (d + 1) + d) % n_strat  # Simple deterministic stratification
                x[d] = lo[d] + (hi[d] - lo[d]) * (stratum + np.random.random()) / n_strat
            f = eval_point(x)
        
        # Add a few more random points to fill up to 80
        while used < 80 and time.time() < deadline:
            x = lo + (hi - lo) * np.random.random(self.dim)
            f = eval_point(x)

        # Now we have ~80 points. Find non-dominated set.
        if not all_F:
            return np.array([])

        all_F_arr = np.array(all_F)
        all_X_arr = np.array(all_X)

        # Filter to only points better than reference point in all objectives
        valid_mask = np.all(all_F_arr < ref_point, axis=1)
        if np.any(valid_mask):
            valid_F = all_F_arr[valid_mask]
            valid_X = all_X_arr[valid_mask]
        else:
            valid_F = all_F_arr
            valid_X = all_X_arr

        # Find non-dominated set from valid points
        if len(valid_F) == 0:
            return all_X_arr

        nd_mask = self._non_dominated_mask(valid_F)
        nd_F = valid_F[nd_mask]
        nd_X = valid_X[nd_mask]

        # Phase 2: Archive-based local search with adaptive step sizes
        # Maintain archive of ND points, generate offspring via mutation
        remaining_budget = self.budget - used
        if remaining_budget > 0 and time.time() < deadline:
            # Start with current ND points as archive
            archive_F = list(nd_F)
            archive_X = list(nd_X)
            
            # Adaptive step size: start large, shrink over time
            total_steps = remaining_budget
            step_idx = 0
            
            while used < self.budget and time.time() < deadline:
                step_idx += 1
                # Linearly decrease step size from 0.15 to 0.01 of range
                progress = min(step_idx / max(total_steps, 1), 1.0)
                base_scale = 0.15 - 0.14 * progress
                scale = base_scale * (hi - lo)
                
                # Pick a random parent from archive (weighted toward hypervolume contributors)
                if len(archive_X) == 0:
                    # No valid points, just do random search
                    x_new = lo + (hi - lo) * np.random.random(self.dim)
                    f_new = eval_point(x_new)
                    if is_valid(f_new):
                        archive_F.append(f_new)
                        archive_X.append(x_new)
                        # Prune dominated
                        self._prune_archive(archive_F, archive_X)
                    continue
                
                # Select parent: prefer points with high hypervolume contribution
                # Simple heuristic: pick randomly, but weight by distance to reference
                n_archive = len(archive_X)
                # Weight by how "central" the point is (closer to reference = better for HV)
                weights = np.array([1.0 / (np.sum((ref_point - f) ** 2) + 1e-10) for f in archive_F])
                weights = weights / np.sum(weights)
                parent_idx = np.random.choice(n_archive, p=weights)
                x_parent = archive_X[parent_idx]
                
                # Generate offspring via Gaussian mutation
                x_new = x_parent + np.random.normal(0, 1, self.dim) * scale
                x_new = np.clip(x_new, lo, hi)
                f_new = eval_point(x_new)
                
                # Check if offspring is valid and non-dominated
                if is_valid(f_new):
                    # Check if it's non-dominated w.r.t. archive
                    dominated = False
                    for af in archive_F:
                        if np.all(af <= f_new) and np.any(af < f_new):
                            dominated = True
                            break
                    if not dominated:
                        archive_F.append(f_new)
                        archive_X.append(x_new)
                        # Remove points dominated by the new point
                        self._prune_archive(archive_F, archive_X)

        # Re-find non-dominated set from all points
        if not all_F:
            return np.array([])

        all_F_arr = np.array(all_F)
        all_X_arr = np.array(all_X)

        # Filter to only points better than reference point in all objectives
        valid_mask = np.all(all_F_arr < ref_point, axis=1)
        if np.any(valid_mask):
            valid_F = all_F_arr[valid_mask]
            valid_X = all_X_arr[valid_mask]
        else:
            valid_F = all_F_arr
            valid_X = all_X_arr

        if len(valid_F) == 0:
            return all_X_arr

        nd_mask = self._non_dominated_mask(valid_F)
        nd_F = valid_F[nd_mask]
        nd_X = valid_X[nd_mask]

        return nd_X

    def _prune_archive(self, archive_F, archive_X):
        """Remove dominated points from the archive (in-place)."""
        n = len(archive_F)
        if n <= 1:
            return
        
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            if not keep[i]:
                continue
            for j in range(n):
                if i == j or not keep[j]:
                    continue
                # j dominates i if j <= i in all objectives and j < i in at least one
                if np.all(archive_F[j] <= archive_F[i]) and np.any(archive_F[j] < archive_F[i]):
                    keep[i] = False
                    break
        
        # Filter in-place
        for i in range(n - 1, -1, -1):
            if not keep[i]:
                del archive_F[i]
                del archive_X[i]

    def _non_dominated_mask(self, F):
        """Return a boolean mask for non-dominated points in F (minimization)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=bool)
        
        nd_mask = np.ones(n, dtype=bool)
        for i in range(n):
            if not nd_mask[i]:
                continue
            for j in range(n):
                if i == j or not nd_mask[j]:
                    continue
                # j dominates i if j <= i in all objectives and j < i in at least one
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nd_mask[i] = False
                    break
        return nd_mask
# EVOLVE-BLOCK-END
