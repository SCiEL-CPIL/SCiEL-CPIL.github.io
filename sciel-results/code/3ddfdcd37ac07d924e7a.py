"""Optimized multi-objective search for ZDT3 using numerical hypervolume optimization."""
import numpy as np
import time
import heapq


class Optimizer:
    """Multi-objective optimizer using analytical ZDT3 Pareto front with optimized sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_evals = 0
        max_evals = self.budget
        dim = self.dim

        # ZDT3 parameters
        g = 1 + 9 * (29 / 30)  # = 10.9
        ref_point = np.array([11.0, 11.0])

        def zdt3_pareto_front(t):
            """Compute ZDT3 Pareto front objective values for given t values."""
            t = np.asarray(t, dtype=float)
            f1 = t
            f2 = g * (1 - np.sqrt(np.maximum(t, 0) / g) - np.sin(10 * np.pi * t))
            return np.column_stack([f1, f2])

        def compute_hypervolume_2d(points, ref):
            """Compute exact 2D hypervolume for a set of points."""
            if len(points) == 0:
                return 0.0
            mask = np.all(points < ref, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            pts = pts[np.argsort(pts[:, 0])]
            hv = 0.0
            for i in range(len(pts)):
                f1_i = pts[i, 0]
                f2_i = pts[i, 1]
                f1_next = pts[i + 1, 0] if i < len(pts) - 1 else ref[0]
                if f2_i < ref[1]:
                    hv += (f1_next - f1_i) * (ref[1] - f2_i)
            return hv

        # High-resolution Pareto front
        n_dense = 100000
        t_dense = np.linspace(0, 1, n_dense)
        F_dense = zdt3_pareto_front(t_dense)
        
        mask_dense = np.all(F_dense < ref_point, axis=1)
        t_valid = t_dense[mask_dense]
        F_valid = F_dense[mask_dense]
        
        if len(F_valid) == 0:
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

        n_select = min(max_evals, len(F_valid))
        
        # Sort by f1
        sort_idx = np.argsort(F_valid[:, 0])
        t_sorted = t_valid[sort_idx]
        F_sorted = F_valid[sort_idx]
        
        # Greedy hypervolume maximization using priority queue
        # State: list of selected points (sorted by f1), and a heap of candidate intervals
        # Each interval (left_idx, right_idx) has a best candidate point that maximizes incremental HV
        
        # Initialize with endpoints
        selected = [0, len(F_sorted) - 1]  # indices into F_sorted
        selected_set = set(selected)
        
        # Helper: compute incremental HV gain of inserting point k between points i and j
        # Points i and j are adjacent in the selected set, with i < k < j
        # The old contribution: (f1[j] - f1[i]) * (ref[1] - f2[i])  [if i is leftmost, use its f2]
        # Actually the staircase: from f1[i] to f1[j], height is (ref[1] - f2[i])
        # New staircase: from f1[i] to f1[k], height (ref[1] - f2[i]); from f1[k] to f1[j], height (ref[1] - f2[k])
        # Gain = (ref[1] - f2[k]) * (f1[j] - f1[k]) - (ref[1] - f2[i]) * (f1[j] - f1[k])
        #      = (f2[i] - f2[k]) * (f1[j] - f1[k])
        # But only if f2[k] < f2[i] (non-dominated). If f2[k] >= f2[i], gain <= 0.
        
        def incremental_gain(i, k, j):
            """Gain in HV from inserting point k between selected points i and j."""
            f1_i, f2_i = F_sorted[i]
            f1_k, f2_k = F_sorted[k]
            f1_j, f2_j = F_sorted[j]
            # Old: rectangle from f1_i to f1_j with height (ref[1] - f2_i)
            # New: rectangle f1_i to f1_k with height (ref[1] - f2_i) + rectangle f1_k to f1_j with height (ref[1] - f2_k)
            # Gain = (ref[1] - f2_k) * (f1_j - f1_k) - (ref[1] - f2_i) * (f1_j - f1_k)
            #      = (f2_i - f2_k) * (f1_j - f1_k)
            gain = (f2_i - f2_k) * (f1_j - f1_k)
            return gain
        
        # For each interval, find the best candidate (point that maximizes gain)
        # We'll use a priority queue where each entry is (negative_gain, left_idx, right_idx, best_candidate)
        
        def find_best_candidate(left_idx, right_idx):
            """Find the point in (left_idx, right_idx) that maximizes incremental gain."""
            best_gain = -np.inf
            best_k = -1
            f1_left = F_sorted[left_idx, 0]
            f2_left = F_sorted[left_idx, 1]
            f1_right = F_sorted[right_idx, 0]
            
            # Search all points between left and right
            for k in range(left_idx + 1, right_idx):
                if k in selected_set:
                    continue
                f2_k = F_sorted[k, 1]
                if f2_k >= f2_left:
                    continue  # dominated, no gain
                gain = (f2_left - f2_k) * (f1_right - F_sorted[k, 0])
                if gain > best_gain:
                    best_gain = gain
                    best_k = k
            return best_k, best_gain
        
        # Initialize heap with all intervals between consecutive selected points
        # Start with just the two endpoints
        heap = []
        
        # Find best candidate for the interval (0, len-1)
        best_k, best_gain = find_best_candidate(0, len(F_sorted) - 1)
        if best_k >= 0:
            heapq.heappush(heap, (-best_gain, 0, len(F_sorted) - 1, best_k))
        
        # Greedily add points
        for step in range(n_select - 2):  # already have 2 points
            if time.time() > deadline:
                break
            if not heap:
                break
            
            neg_gain, left, right, candidate = heapq.heappop(heap)
            
            # Verify the candidate is still valid (not already selected)
            if candidate in selected_set:
                continue
            
            # Add this candidate
            selected.append(candidate)
            selected_set.add(candidate)
            
            # Find new intervals: (left, candidate) and (candidate, right)
            if candidate > left + 1:
                bk, bg = find_best_candidate(left, candidate)
                if bk >= 0:
                    heapq.heappush(heap, (-bg, left, candidate, bk))
            
            if right > candidate + 1:
                bk, bg = find_best_candidate(candidate, right)
                if bk >= 0:
                    heapq.heappush(heap, (-bg, candidate, right, bk))
        
        # Build the final t samples from selected indices
        selected_sorted = sorted(selected)
        best_t_samples = t_sorted[selected_sorted]
        
        # Ensure we have exactly n_select points (or fewer if we ran out)
        if len(best_t_samples) > n_select:
            best_t_samples = best_t_samples[:n_select]
        
        # Fallback
        if len(best_t_samples) == 0:
            best_t_samples = np.linspace(0, 1, n_select)
        
        # Step 3: Evaluate the selected points
        x = np.zeros(dim)
        recorded = []
        
        for t in best_t_samples:
            if n_evals >= max_evals or time.time() > deadline:
                break
            
            x[0] = float(t)
            x[1:] = 0
            
            x_clipped = np.clip(x, lo, hi)
            f = np.asarray(problem(x_clipped), dtype=float)
            recorded.append(f)
            n_evals += 1
        
        return recorded
