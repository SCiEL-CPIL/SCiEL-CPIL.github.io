"""Optimized multi-objective optimizer for RE21 with hypervolume-guided adaptive DE."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Hypervolume-guided adaptive differential evolution for 2-objective RE21."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget
        ref = np.array([3000.0, 0.06])
        
        # Helper: Check if point a dominates point b (a is better)
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        # Helper: Calculate hypervolume contribution of each point in a non-dominated set
        # Assumes points are 2D and non-dominated
        def hv_contributions(points, ref):
            """
            Returns an array of HV contributions for each point in the non-dominated set.
            Points must be non-dominated.
            """
            if len(points) == 0:
                return np.array([])
            
            # Sort by f1 ascending
            order = np.argsort(points[:, 0])
            pts = points[order]
            
            contributions = np.zeros(len(pts))
            min_y = ref[1]
            
            # We need to map back to original indices? 
            # For pruning, we just need to know which one to remove.
            # We can return contributions aligned with the sorted order, 
            # but we need to identify the point in the original list.
            # Let's return the contribution for each point in the sorted list,
            # and also the original indices.
            
            # Actually, easier: just compute the total HV and then remove the point
            # that minimizes the HV drop. 
            # But that is O(N^2). 
            # For N < 50, it's fine.
            # However, the "slice" method gives a direct contribution metric.
            
            # Slice method:
            # Sort by x asc. y is desc (for non-dominated).
            # Contribution of i is (x_{i+1} - x_i) * (y_i - y_{i+1})? 
            # No, the area added by point i is the rectangle from x_i to x_{i+1} 
            # with height (current_min_y - y_i).
            
            # Let's stick to the simple O(N^2) pruning if N is small, 
            # or use the slice approximation.
            
            # Slice approximation:
            # For i in range(len(pts)):
            #   next_x = pts[i+1, 0] if i+1 < len(pts) else ref[0]
            #   height = min_y - pts[i, 1]
            #   contrib[i] = (next_x - pts[i, 0]) * height
            #   min_y = pts[i, 1]
            
            # This is not strictly the marginal contribution if points are not "convex", 
            # but it's a good heuristic.
            
            # Let's use the exact marginal contribution by recalculating HV?
            # No, let's use the slice heuristic for speed.
            
            for i in range(len(pts)):
                x_i, y_i = pts[i]
                next_x = pts[i+1, 0] if i+1 < len(pts) else ref[0]
                # If y_i is not less than min_y, it's dominated by previous in y? 
                # But input is non-dominated, so y_i should be < min_y (which is min y of previous).
                # Wait, if we sort by x asc, and they are non-dominated, y must be desc.
                # So y_i < min_y (where min_y is the y of the previous point, or ref).
                
                height = min_y - y_i
                width = next_x - x_i
                contributions[i] = width * height
                min_y = y_i
                
            return contributions, order

        # Initialize archive
        # Use random initialization with np.random (respects seed)
        n_init = 50
        X_init = lo + (hi - lo) * np.random.rand(n_init, dim)
        F_init = np.array([problem(x) for x in X_init])
        used = n_init
        
        # Build initial archive
        archive_X = []
        archive_F = []
        
        for i in range(n_init):
            # Check if dominated by existing archive
            is_dominated = False
            for j in range(len(archive_F)):
                if dominates(archive_F[j], F_init[i]):
                    is_dominated = True
                    break
            if is_dominated:
                continue
                
            # Remove points dominated by new point
            new_archive_X = []
            new_archive_F = []
            for j in range(len(archive_F)):
                if not dominates(F_init[i], archive_F[j]):
                    new_archive_X.append(archive_X[j])
                    new_archive_F.append(archive_F[j])
            
            new_archive_X.append(X_init[i])
            new_archive_F.append(F_init[i])
            archive_X = new_archive_X
            archive_F = new_archive_F
            
            # Prune if too large
            if len(archive_F) > 30:
                # Convert to numpy for HV contribution calc
                F_arr = np.array(archive_F)
                contribs, order = hv_contributions(F_arr, ref)
                # Find index with min contribution
                min_contrib_idx = np.argmin(contribs)
                # The order maps sorted index to original index? 
                # My hv_contributions returns contributions in sorted order.
                # We need to remove the point in archive_F that corresponds to this.
                # The 'order' array contains the original indices in the sorted sequence.
                # So the point to remove is archive_F[order[min_contrib_idx]]
                idx_to_remove = order[min_contrib_idx]
                archive_X.pop(idx_to_remove)
                archive_F.pop(idx_to_remove)

        # Main loop
        while used < budget:
            # Adaptive mutation factor
            progress = used / budget
            F_mut = 0.5 + 0.5 * (1.0 - progress) # Start 1.0, end 0.5? Or 0.8 to 0.2?
            # Let's use 0.8 to 0.3
            F_mut = 0.8 - 0.5 * progress
            
            # Select parents from archive
            n_archive = len(archive_F)
            if n_archive == 0:
                # Fallback to random
                x_parent = lo + (hi - lo) * np.random.rand(dim)
            elif n_archive == 1:
                x_parent = archive_X[0]
            else:
                # Pick 3 distinct parents
                idxs = np.random.choice(n_archive, size=3, replace=False)
                p1 = archive_X[idxs[0]]
                p2 = archive_X[idxs[1]]
                p3 = archive_X[idxs[2]]
                x_parent = p1 + F_mut * (p2 - p3)
                
            # Add perturbation
            x_child = x_parent + 0.1 * (hi - lo) * np.random.standard_normal(dim)
            x_child = np.clip(x_child, lo, hi)
            
            f_child = problem(x_child)
            used += 1
            
            # Update archive
            # Check if dominated
            is_dominated = False
            for j in range(len(archive_F)):
                if dominates(archive_F[j], f_child):
                    is_dominated = True
                    break
            
            if not is_dominated:
                # Remove dominated
                new_archive_X = []
                new_archive_F = []
                for j in range(len(archive_F)):
                    if not dominates(f_child, archive_F[j]):
                        new_archive_X.append(archive_X[j])
                        new_archive_F.append(archive_F[j])
                
                new_archive_X.append(x_child)
                new_archive_F.append(f_child)
                archive_X = new_archive_X
                archive_F = new_archive_F
                
                # Prune
                if len(archive_F) > 30:
                    F_arr = np.array(archive_F)
                    contribs, order = hv_contributions(F_arr, ref)
                    min_contrib_idx = np.argmin(contribs)
                    idx_to_remove = order[min_contrib_idx]
                    archive_X.pop(idx_to_remove)
                    archive_F.pop(idx_to_remove)
        
        # Return any representative, e.g., the best point or the archive
        # The return value is ignored, but we must return something.
        if archive_X:
            return np.array(archive_X)
        else:
            return np.array([lo + (hi - lo) * 0.5])

# EVOLVE-BLOCK-END
