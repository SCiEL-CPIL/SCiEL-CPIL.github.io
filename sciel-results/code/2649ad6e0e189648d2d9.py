"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective optimizer for WFG4 with NSGA-II style selection and coordinate descent polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated(self, F):
        """Return indices of non-dominated points."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
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

    def _crowding_distance(self, F):
        """Compute crowding distance for each point in F."""
        n = len(F)
        if n == 0:
            return np.array([])
        if n <= 2:
            return np.full(n, np.inf)
        
        dist = np.zeros(n)
        for m in range(self.n_obj):
            order = np.argsort(F[:, m])
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            f_range = F[order[-1], m] - F[order[0], m]
            if f_range > 0:
                for i in range(1, n - 1):
                    dist[order[i]] += (F[order[i + 1], m] - F[order[i - 1], m]) / f_range
        return dist

    def _hypervolume_contribution(self, point_F, archive_F, ref_point):
        """Compute the hypervolume contribution of a single point given an archive."""
        # Only count if point is strictly better than reference in all objectives
        if not np.all(point_F < ref_point):
            return 0.0
        
        # Check if point is dominated by any point in archive
        for af in archive_F:
            if np.all(af <= point_F) and np.any(af < point_F):
                return 0.0
        
        # Compute contribution: for each objective, the "box" this point extends
        # This is a simplified contribution: the volume of the region dominated by this point
        # but not dominated by any point in the archive.
        # For 3 objectives, we can compute this exactly.
        # The contribution is the volume of the box [point_F, ref_point] minus overlaps with archive points.
        
        # Simple approach: compute the volume of the box defined by point_F and ref_point
        # then subtract volumes of sub-boxes that are already covered.
        # For simplicity and speed, use a basic approximation or exact 3D calculation.
        
        # Exact 3D hypervolume contribution:
        # The point contributes the volume of the region R = {y : point_F <= y < ref_point}
        # that is not dominated by any point in archive_F.
        
        # Since we already checked it's not dominated, we need to subtract the parts of R
        # that are dominated by archive points.
        
        # For 3 objectives, this can be done by sorting and inclusion-exclusion,
        # but for speed, we'll use a simple lower bound: the volume of the box
        # [point_F, ref_point] is a lower bound on the contribution.
        # Actually, the contribution is at most this volume.
        
        # Let's use a more accurate method: the contribution is the volume of R
        # minus the volume of R ∩ (union of boxes dominated by archive points).
        
        # For efficiency, we'll approximate by just returning the box volume as an upper bound,
        # which is good enough for guiding the coordinate descent.
        
        box_volume = 1.0
        for m in range(self.n_obj):
            box_volume *= (ref_point[m] - point_F[m])
        
        return box_volume

    def _coordinate_descent_polish(self, x, F, problem, lo, hi, budget_left, ref_point, archive_F):
        """Perform coordinate descent to improve the point, using hypervolume contribution as guide."""
        best_x = x.copy()
        best_F = F.copy()
        best_hv = self._hypervolume_contribution(F, archive_F, ref_point)
        
        # Try small perturbations in each coordinate
        for coord in range(self.dim):
            if budget_left <= 0:
                break
            
            # Try a few small steps in each direction
            for delta in [0.01, 0.05, 0.1, -0.01, -0.05, -0.1]:
                if budget_left <= 0:
                    break
                
                trial_x = best_x.copy()
                trial_x[coord] += delta * (hi[coord] - lo[coord])
                
                if trial_x[coord] < lo[coord] or trial_x[coord] > hi[coord]:
                    continue
                
                trial_F = problem(trial_x)
                budget_left -= 1
                
                trial_hv = self._hypervolume_contribution(trial_F, archive_F, ref_point)
                
                if trial_hv > best_hv:
                    best_x = trial_x
                    best_F = trial_F
                    best_hv = trial_hv
        
        return best_x, best_F, budget_left

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(0)

        # Initial population
        pop_size = min(40, self.budget // 5)
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Reference point for hypervolume
        ref_point = np.array([3.0, 5.0, 7.0])

        # Archive of all non-dominated points found so far
        archive_F = F.copy()

        # Adaptive mutation parameters
        base_scale = 0.25
        min_scale = 0.01

        # Phase 1: Evolutionary search
        while used < self.budget * 0.7:
            if time.time() > deadline:
                break

            # Get non-dominated front from current population
            nd_indices = self._fast_non_dominated(F)
            if len(nd_indices) == 0:
                nd_indices = np.array([np.argmin(np.sum(F, axis=1))])

            # Compute crowding distance for the non-dominated front
            nd_F = F[nd_indices]
            cd = self._crowding_distance(nd_F)

            # Select parent: prefer points with higher crowding distance
            if len(nd_indices) >= 2:
                i = nd_indices[rng.integers(len(nd_indices))]
                j = nd_indices[rng.integers(len(nd_indices))]
                i_pos = np.where(nd_indices == i)[0][0]
                j_pos = np.where(nd_indices == j)[0][0]
                if cd[i_pos] >= cd[j_pos]:
                    parent_idx = i
                else:
                    parent_idx = j
            else:
                parent_idx = nd_indices[0]

            # Adaptive mutation
            progress = used / (self.budget * 0.7)
            scale = base_scale * (1 - progress) + min_scale * progress

            # Gaussian mutation
            mutation = scale * (hi - lo) * rng.standard_normal(self.dim)
            child = X[parent_idx] + mutation
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Update population: replace worst individual
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = child
            F[worst_idx] = fc

            # Update archive
            archive_F = np.vstack([archive_F, fc.reshape(1, -1)])

        # Phase 2: Coordinate descent polish on non-dominated points
        if time.time() < deadline and used < self.budget:
            nd_indices = self._fast_non_dominated(F)
            budget_left = self.budget - used
            
            # Polish the top non-dominated points
            for idx in nd_indices[:min(10, len(nd_indices))]:
                if budget_left <= 0 or time.time() > deadline:
                    break
                
                polished_x, polished_F, budget_left = self._coordinate_descent_polish(
                    X[idx], F[idx], problem, lo, hi, budget_left, ref_point, archive_F
                )
                used = self.budget - budget_left
                
                if np.any(polished_F < F[idx]):
                    X[idx] = polished_x
                    F[idx] = polished_F
                    archive_F = np.vstack([archive_F, polished_F.reshape(1, -1)])

        return X
# EVOLVE-BLOCK-END
