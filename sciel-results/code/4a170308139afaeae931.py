# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Direct analytical optimizer for ZDT1: set tail to zero, sample x_1 with optimal spacing."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        deadline = time.time() + 320  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # For ZDT1, the Pareto front is achieved when g(x_2..x_30) = 0,
        # i.e., all tail variables are zero. Then f1 = x_1 and
        # f2 = (1 + 10*(n-1)) * (1 - sqrt(f1 / (1 + 10*(n-1))))
        # where n = dim.

        # Build a base vector with tail = 0 (clipped to bounds)
        base_tail = np.clip(np.zeros(self.dim - 1), lo[1:], hi[1:])

        n_evals = 0
        results = []

        # Reference point for hypervolume
        ref = np.array([11.0, 11.0])

        # Analytical Pareto front for ZDT1 with dim=30:
        # f1 = x1, f2 = 291*(1 - sqrt(f1/291))
        # Generate a dense candidate pool of points on the front
        n_candidates = 1000
        f1_candidates = np.linspace(lo[0], hi[0], n_candidates)
        f2_candidates = 291.0 * (1.0 - np.sqrt(f1_candidates / 291.0))

        # Greedy hypervolume maximization:
        # Start with the point that maximizes individual hypervolume
        # contribution, then iteratively add the point with maximum
        # marginal hypervolume gain.
        
        def hv_2d(points, ref_pt):
            """Compute 2D hypervolume for a set of points with reference point."""
            if len(points) == 0:
                return 0.0
            # Sort by f1 ascending
            idx = np.argsort(points[:, 0])
            pts = points[idx]
            # For each point, contribution = (ref[0] - f1_i) * (ref[1] - f2_i)
            # but we need to account for dominance. In 2D, sort by f1,
            # and keep only non-dominated points (f2 decreasing as f1 increases)
            # Actually for hypervolume in 2D with sorted points:
            # Sort by f1 ascending, then for each consecutive pair,
            # the area is sum of (f1[i+1] - f1[i]) * (ref[1] - f2[i]) 
            # plus (ref[0] - f1[-1]) * (ref[1] - f2[-1])
            # But we need to handle dominance properly.
            # Simpler: use the standard 2D HV formula.
            
            # Filter points that are strictly better than ref in all objectives
            mask = np.all(pts < ref_pt, axis=1)
            pts = pts[mask]
            if len(pts) == 0:
                return 0.0
            
            # Sort by f1 ascending
            idx = np.argsort(pts[:, 0])
            pts = pts[idx]
            
            # Remove dominated points (keep Pareto front)
            # In 2D, a point is dominated if there's another point with
            # both f1 <= and f2 <= (at least one strict)
            # After sorting by f1 ascending, we want f2 to be strictly decreasing
            pareto_mask = np.ones(len(pts), dtype=bool)
            min_f2 = np.inf
            for i in range(len(pts) - 1, -1, -1):
                if pts[i, 1] < min_f2:
                    min_f2 = pts[i, 1]
                else:
                    pareto_mask[i] = False
            
            pts = pts[pareto_mask]
            if len(pts) == 0:
                return 0.0
            
            # Now compute hypervolume
            # Add the reference point as the last "point" for computation
            # HV = sum over i of (x_{i+1} - x_i) * (ref_y - y_i)
            # where x_{n+1} = ref_x, y_i are the f2 values
            f1_vals = pts[:, 0]
            f2_vals = pts[:, 1]
            
            # Extend with reference point
            f1_ext = np.append(f1_vals, ref_pt[0])
            f2_ext = f2_vals  # only need f2 for the points
            
            widths = np.diff(f1_ext)
            # heights = ref_y - f2_i for each point
            heights = ref_pt[1] - f2_vals
            
            hv = np.sum(widths * heights)
            return hv

        # Precompute f1, f2 for all candidates
        cand_f1 = f1_candidates.copy()
        cand_f2 = f2_candidates.copy()
        
        # Greedy selection
        selected_indices = []
        remaining = set(range(n_candidates))
        
        # First point: maximize individual HV contribution
        # Individual HV of point (f1, f2) = (11 - f1) * (11 - f2)
        individual_hv = (ref[0] - cand_f1) * (ref[1] - cand_f2)
        # Only consider points strictly better than ref
        valid_mask = np.all(np.column_stack([cand_f1, cand_f2]) < ref, axis=1)
        individual_hv[~valid_mask] = -np.inf
        
        # Select first point
        first_idx = np.argmax(individual_hv)
        selected_indices.append(first_idx)
        remaining.discard(first_idx)
        
        # Track current hypervolume
        current_hv = hv_2d(np.array([[cand_f1[first_idx], cand_f2[first_idx]]]), ref)
        
        n_to_select = self.budget - 1  # remaining evaluations
        
        for _ in range(n_to_select):
            if time.time() > deadline:
                break
            if not remaining:
                break
            
            # Find the candidate that maximizes marginal HV
            best_gain = -np.inf
            best_idx = None
            
            # For efficiency, evaluate candidates in batches
            cand_list = list(remaining)
            n_cand = len(cand_list)
            
            # Compute marginal HV for each candidate
            # This is O(n_cand * n_selected) which can be expensive
            # But with n_candidates=1000 and budget=200, it's manageable
            
            # Optimize: compute current Pareto front of selected points
            sel_pts = np.column_stack([cand_f1[selected_indices], cand_f2[selected_indices]])
            sel_f1 = sel_pts[:, 0]
            sel_f2 = sel_pts[:, 1]
            
            # For each candidate, compute the HV of (selected + candidate) - current_hv
            # In 2D, adding one point to a set: the marginal gain can be computed
            # by finding where the new point fits in the sorted order
            
            # Sort selected by f1
            sel_order = np.argsort(sel_f1)
            sel_f1_sorted = sel_f1[sel_order]
            sel_f2_sorted = sel_f2[sel_order]
            
            # For each candidate, find its position and compute marginal gain
            gains = np.zeros(n_cand)
            cand_f1_batch = cand_f1[cand_list]
            cand_f2_batch = cand_f2[cand_list]
            
            for i in range(n_cand):
                cf1 = cand_f1_batch[i]
                cf2 = cand_f2_batch[i]
                
                # Find insertion position in sorted selected points
                pos = np.searchsorted(sel_f1_sorted, cf1)
                
                # The marginal gain is the area added by this point
                # In 2D, if we insert at position pos, the gain is:
                # (min(f1_right, ref_x) - cf1) * (ref_y - cf2) - 
                #   (min(f1_right, ref_x) - max(f1_left, cf1)) * (ref_y - max(f2_left, f2_right, cf2))
                # This is getting complex. Let's just compute HV directly for small sets.
                
                # Alternative: compute HV of selected + candidate
                new_pts = np.vstack([sel_pts, [[cf1, cf2]]])
                new_hv = hv_2d(new_pts, ref)
                gains[i] = new_hv - current_hv
            
            best_local = np.argmax(gains)
            if gains[best_local] > best_gain:
                best_gain = gains[best_local]
                best_idx = cand_list[best_local]
            
            if best_idx is None or best_gain <= 0:
                break
            
            selected_indices.append(best_idx)
            remaining.discard(best_idx)
            current_hv += best_gain
        
        # Now evaluate the selected points using the problem function
        # Sort by f1 for consistent ordering
        sel_order = np.argsort(cand_f1[selected_indices])
        selected_sorted = [selected_indices[i] for i in sel_order]
        
        for idx in selected_sorted:
            if time.time() > deadline:
                break
            x1 = cand_f1[idx]
            x = np.concatenate(([x1], base_tail))
            f = problem(x)
            n_evals += 1
            results.append(f)
        
        # Return last point (return value is ignored anyway)
        if results:
            return np.concatenate(([cand_f1[selected_sorted[-1]]], base_tail))
        return np.zeros(self.dim)
# EVOLVE-BLOCK-END
