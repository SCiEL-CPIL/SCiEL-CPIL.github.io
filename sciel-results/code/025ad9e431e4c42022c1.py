"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective search with adaptive mutation and hypervolume-aware replacement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # Population size
        pop_size = int(min(40, self.budget // 5))
        if pop_size < 10:
            pop_size = 10

        # Random initialization
        X = lo + range_ * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Reference point for hypervolume scoring
        ref_point = np.array([3.0, 5.0, 7.0])

        def get_non_dominated_indices(F_mat):
            """Return indices of non-dominated points in F_mat."""
            n = len(F_mat)
            non_dominated = np.ones(n, dtype=bool)
            for i in range(n):
                if not non_dominated[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_mat[j] <= F_mat[i]) and np.any(F_mat[j] < F_mat[i]):
                        non_dominated[i] = False
                        break
            return np.where(non_dominated)[0]

        def compute_hv_contribution(point, F_mat):
            """Compute hypervolume contribution of a point using inclusion-exclusion for 3D."""
            if np.any(point >= ref_point):
                return 0.0
            
            # For 3 objectives, we can compute exact contribution
            # The contribution is the volume of the box [point, ref_point] minus
            # volumes already covered by other non-dominated points
            
            # Get all non-dominated points including this one
            nd_idx = get_non_dominated_indices(F_mat)
            nd_points = F_mat[nd_idx]
            
            # Find the index of our point in nd_points
            point_idx = -1
            for i, p in enumerate(nd_points):
                if np.allclose(p, point):
                    point_idx = i
                    break
            
            if point_idx == -1:
                # Point is dominated, contribution is 0
                return 0.0
            
            # Compute contribution using the standard 3D HV algorithm
            # Sort by first objective
            points = nd_points.copy()
            n = len(points)
            
            if n == 0:
                return 0.0
            
            # Sort by first coordinate
            order = np.argsort(points[:, 0])
            sorted_points = points[order]
            
            # Binary search based approach for 3D HV
            # For each point, compute the volume it uniquely contributes
            
            # Simple O(n^2) approach for small n
            contributions = np.zeros(n)
            for i in range(n):
                p = sorted_points[i]
                if np.any(p >= ref_point):
                    continue
                
                # Volume of box from p to ref
                box_vol = np.prod(ref_point - p)
                
                # Subtract volumes covered by other points that dominate in the relevant region
                covered = 0.0
                for j in range(n):
                    if i == j:
                        continue
                    q = sorted_points[j]
                    if np.any(q >= ref_point):
                        continue
                    
                    # Check if q covers part of p's box
                    # q covers p's box in dimensions where q[k] < p[k]
                    # The intersection of [p, ref] and [q, ref] is [max(p,q), ref]
                    lower = np.maximum(p, q)
                    if np.all(lower < ref_point):
                        covered += np.prod(ref_point - lower)
                
                # But this double-counts; we need a proper inclusion-exclusion
                # For simplicity, use a heuristic: if another point dominates p in all dims, contribution is 0
                dominated_by_other = False
                for j in range(n):
                    if i == j:
                        continue
                    q = sorted_points[j]
                    if np.all(q <= p) and np.any(q < p):
                        dominated_by_other = True
                        break
                
                if dominated_by_other:
                    contributions[i] = 0.0
                else:
                    # Approximate: use the box volume minus average overlap
                    # This is not exact but gives a reasonable ranking
                    contributions[i] = box_vol
            
            # Find the contribution of our specific point
            if point_idx >= 0 and point_idx < n:
                # Map back to original index
                orig_idx = order[point_idx]
                return contributions[point_idx]
            
            return 0.0

        def compute_hv_contributions_batch(F_mat):
            """Compute hypervolume contributions for all points in F_mat."""
            n = len(F_mat)
            contributions = np.zeros(n)
            
            # Filter to non-dominated points
            nd_mask = np.zeros(n, dtype=bool)
            for i in range(n):
                is_dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_mat[j] <= F_mat[i]) and np.any(F_mat[j] < F_mat[i]):
                        is_dominated = True
                        break
                if not is_dominated:
                    nd_mask[i] = True
            
            nd_indices = np.where(nd_mask)[0]
            if len(nd_indices) == 0:
                return contributions
            
            nd_points = F_mat[nd_indices]
            m = len(nd_points)
            
            # For each non-dominated point, compute its contribution
            # Using a simple O(m^2) approach
            nd_contribs = np.zeros(m)
            
            for i in range(m):
                p = nd_points[i]
                if np.any(p >= ref_point):
                    nd_contribs[i] = 0.0
                    continue
                
                box_vol = np.prod(ref_point - p)
                
                # Check if any other point dominates
                dominated = False
                for j in range(m):
                    if i == j:
                        continue
                    q = nd_points[j]
                    if np.all(q <= p) and np.any(q < p):
                        dominated = True
                        break
                
                if dominated:
                    nd_contribs[i] = 0.0
                else:
                    # Approximate contribution
                    nd_contribs[i] = box_vol
            
            for i, idx in enumerate(nd_indices):
                contributions[idx] = nd_contribs[i]
            
            return contributions

        # Main evolution loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Get non-dominated parents
            nd_idx = get_non_dominated_indices(F)
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(F))
            
            parents = X[nd_idx]
            F_parents = F[nd_idx]
            
            # Select parent: prefer points with higher HV contribution
            # Compute contributions for non-dominated points
            parent_contribs = compute_hv_contributions_batch(F_parents)
            
            # Use tournament selection based on HV contribution
            if np.any(parent_contribs > 0):
                # Normalize and use as selection weights
                weights = parent_contribs + 1e-10
                weights = weights / weights.sum()
                parent_idx = rng.choice(len(parents), p=weights)
            else:
                parent_idx = rng.integers(len(parents))
            
            parent = parents[parent_idx]
            
            # Adaptive mutation step size
            progress = used / self.budget
            base_step = 0.25 * (1 - progress) + 0.03
            
            # Occasionally use a larger step for exploration
            if rng.random() < 0.08:
                step_size = base_step * 2.5
            else:
                step_size = base_step
            
            # Generate child
            child = parent + step_size * range_ * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Determine which point to replace
            dominated_mask = np.array([
                np.all(F[i] >= fc) and np.any(F[i] > fc) for i in range(len(F))
            ])
            
            if np.any(dominated_mask):
                dominated_indices = np.where(dominated_mask)[0]
                replace_idx = dominated_indices[rng.integers(len(dominated_indices))]
            else:
                # Replace the point with the lowest HV contribution among non-dominated
                # or the worst if all are dominated
                contribs = compute_hv_contributions_batch(F)
                
                # Among non-dominated, pick the one with lowest contribution
                nd_idx_all = get_non_dominated_indices(F)
                if len(nd_idx_all) > 0:
                    nd_contribs = contribs[nd_idx_all]
                    min_contrib_idx = nd_idx_all[np.argmin(nd_contribs)]
                    replace_idx = min_contrib_idx
                else:
                    # All dominated, replace the worst
                    sums = np.sum(F, axis=1)
                    replace_idx = int(np.argmax(sums))
                
                # Only accept if child is not strictly worse
                if np.all(fc >= F[replace_idx]) and np.any(fc > F[replace_idx]):
                    # Child is dominated by the point we're replacing, skip
                    continue

            X[replace_idx] = child
            F[replace_idx] = fc
        
        # Final local refinement pass: use remaining budget for fine-tuning
        remaining = self.budget - used
        if remaining > 5 and time.time() < deadline - 30:
            # Get non-dominated points
            nd_idx = get_non_dominated_indices(F)
            if len(nd_idx) > 0:
                # Do local search around best points
                for _ in range(min(remaining, 20)):
                    if time.time() > deadline:
                        break
                    
                    # Pick a non-dominated point
                    idx = nd_idx[rng.integers(len(nd_idx))]
                    base_point = X[idx]
                    
                    # Small perturbation
                    small_step = 0.02 * range_
                    child = base_point + small_step * rng.standard_normal(self.dim)
                    child = np.clip(child, lo, hi)
                    
                    fc = problem(child)
                    used += 1
                    
                    # Replace if it dominates or has better HV contribution
                    dominated_mask = np.array([
                        np.all(F[i] >= fc) and np.any(F[i] > fc) for i in range(len(F))
                    ])
                    
                    if np.any(dominated_mask):
                        dominated_indices = np.where(dominated_mask)[0]
                        replace_idx = dominated_indices[rng.integers(len(dominated_indices))]
                        X[replace_idx] = child
                        F[replace_idx] = fc
                    else:
                        # Check if it improves the population
                        contribs = compute_hv_contributions_batch(F)
                        nd_idx_all = get_non_dominated_indices(F)
                        if len(nd_idx_all) > 0:
                            nd_contribs = contribs[nd_idx_all]
                            min_contrib_idx = nd_idx_all[np.argmin(nd_contribs)]
                            # Only replace if child is not worse
                            if not (np.all(fc >= F[min_contrib_idx]) and np.any(fc > F[min_contrib_idx])):
                                X[min_contrib_idx] = child
                                F[min_contrib_idx] = fc
        
        return X
# EVOLVE-BLOCK-END
