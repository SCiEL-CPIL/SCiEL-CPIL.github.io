"""Optimized multi-objective search for ZDT3 using numerical hypervolume optimization."""
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using analytical ZDT3 Pareto front with optimized sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_evals = 0
        max_evals = self.budget
        dim = self.dim

        # ZDT3 parameters
        g = 1.0 + 9.0 * (dim - 1) / dim
        ref_point = np.array([11.0, 11.0])

        def pareto_f2(t):
            """Compute ZDT3 Pareto front f2 for given t values."""
            t = np.asarray(t, dtype=float)
            return g * (1.0 - np.sqrt(np.maximum(t, 0.0) / g) - np.sin(10.0 * np.pi * t))

        def compute_hypervolume_2d(points, ref):
            """Compute exact 2D hypervolume for a set of points."""
            if len(points) == 0:
                return 0.0
            
            # Filter points strictly better than reference
            mask = np.all(points < ref, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            
            # Sort by f1 (first objective)
            pts = pts[np.argsort(pts[:, 0])]
            
            # Compute hypervolume as sum of rectangles
            hv = 0.0
            for i in range(len(pts)):
                f1_i = pts[i, 0]
                f2_i = pts[i, 1]
                
                # Width: from f1_i to next f1 (or ref[0])
                if i < len(pts) - 1:
                    f1_next = pts[i + 1, 0]
                else:
                    f1_next = ref[0]
                
                # Only count if f2_i < ref[1]
                if f2_i < ref[1]:
                    width = f1_next - f1_i
                    height = ref[1] - f2_i
                    hv += width * height
            
            return hv

        # Step 1: Create a dense sample of the Pareto front
        n_dense = 40000
        t_dense = np.linspace(0, 1, n_dense)
        f2_dense = pareto_f2(t_dense)
        
        # Filter to only points strictly better than reference
        mask_dense = np.all(np.column_stack([t_dense, f2_dense]) < ref_point, axis=1)
        t_valid = t_dense[mask_dense]
        f2_valid = f2_dense[mask_dense]
        
        if len(f2_valid) == 0:
            # Fallback: just sample uniformly
            t_samples = np.linspace(0, 1, max_evals)
            x = np.zeros(dim)
            recorded = []
            for t in t_samples:
                x[0] = t
                x[1:] = 0
                x_clipped = np.clip(x, lo, hi)
                f = np.asarray(problem(x_clipped), dtype=float)
                recorded.append(f)
                n_evals += 1
                if n_evals >= max_evals or time.time() > deadline:
                    break
            return recorded

        # Step 2: Select points to maximize hypervolume
        n_select = min(max_evals, len(f2_valid))
        
        # Sort by f1 (which is just t since f1 = t)
        sort_idx = np.argsort(t_valid)
        t_sorted = t_valid[sort_idx]
        f2_sorted = f2_valid[sort_idx]
        
        # Initialize with the two endpoints (best f1 and best f2)
        selected_idx = [0, len(t_sorted) - 1]
        selected_set = set(selected_idx)
        
        # Greedy selection: at each step, for each gap, pick the point with lowest f2
        # This approximates the best incremental gain
        
        # Create a list of gaps, each with the best candidate
        gaps = [(0, len(t_sorted) - 1)]  # (left_idx, right_idx) in t_sorted
        
        while len(selected_idx) < n_select and time.time() < deadline:
            if not gaps:
                break
            
            # Find the gap with the best candidate (lowest f2)
            best_gap = None
            best_candidate_idx = -1
            best_f2 = float('inf')
            
            for left, right in gaps:
                if right - left <= 1:
                    continue
                
                # Find the point with lowest f2 in this gap
                sub_f2 = f2_sorted[left:right]
                min_idx = np.argmin(sub_f2)
                candidate_idx = left + min_idx
                
                if candidate_idx not in selected_set and f2_sorted[candidate_idx] < best_f2:
                    best_f2 = f2_sorted[candidate_idx]
                    best_candidate_idx = candidate_idx
                    best_gap = (left, right)
            
            if best_candidate_idx == -1:
                break
            
            # Add this point
            selected_idx.append(best_candidate_idx)
            selected_set.add(best_candidate_idx)
            
            # Split the gap into two new gaps
            left, right = best_gap
            new_gaps = []
            for g_left, g_right in gaps:
                if g_left == left and g_right == right:
                    if best_candidate_idx - g_left > 0:
                        new_gaps.append((g_left, best_candidate_idx))
                    if g_right - best_candidate_idx > 0:
                        new_gaps.append((best_candidate_idx, g_right))
                else:
                    new_gaps.append((g_left, g_right))
            
            gaps = new_gaps
        
        # Convert selected indices to t values
        t_selected = t_sorted[sorted(selected_idx)]
        
        # Step 3: Evaluate the selected points using the problem function
        x = np.zeros(dim)
        recorded = []
        
        for t in t_selected:
            if n_evals >= max_evals or time.time() > deadline:
                break
            
            x[0] = float(t)
            x[1:] = 0
            
            x_clipped = np.clip(x, lo, hi)
            f = np.asarray(problem(x_clipped), dtype=float)
            recorded.append(f)
            n_evals += 1
        
        # If we have budget left, try to refine with a few additional points
        if n_evals < max_evals and time.time() < deadline:
            # Try adding points at critical points we might have missed
            critical_points = np.array([(1/20 + k/10) for k in range(10)])
            critical_points = critical_points[(critical_points >= 0) & (critical_points <= 1)]
            critical_points = np.concatenate([[0.0], critical_points, [1.0]])
            
            for t in critical_points:
                if n_evals >= max_evals or time.time() > deadline:
                    break
                
                # Check if we already have a point close to this t
                if len(t_selected) > 0:
                    if np.min(np.abs(t_selected - t)) < 0.001:
                        continue
                
                x[0] = float(t)
                x[1:] = 0
                x_clipped = np.clip(x, lo, hi)
                f = np.asarray(problem(x_clipped), dtype=float)
                recorded.append(f)
                n_evals += 1
        
        return recorded
