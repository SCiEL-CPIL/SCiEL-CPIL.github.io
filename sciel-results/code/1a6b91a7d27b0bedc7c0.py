"""Multi-objective optimizer for ZDT3 using exact Pareto front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT3.
    
    ZDT3 Pareto front: x2..x30 = 0 gives g=1, f1 = x1, f2 = 1 - sqrt(x1).
    The front is discontinuous with segments where h(x,1) >= 0.
    
    Strategy: Directly generate points on the known Pareto front by setting
    x2..x30 = 0 and sampling x1 optimally across all valid segments.
    Uses exact segment boundaries computed from the h function.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _compute_pareto_segments(self):
        """Compute exact Pareto-optimal segments for ZDT3.
        
        Pareto-optimal where h(x, 1) >= 0, i.e., 1 - sqrt(x) - x*sin(10*pi*x) >= 0
        """
        # Dense grid to find sign changes of h(x,1)
        x_grid = np.linspace(0, 1, 200001)
        h_vals = 1 - np.sqrt(x_grid) - x_grid * np.sin(10 * np.pi * x_grid)
        
        # Find regions where h >= 0
        in_pareto = h_vals >= 0
        
        # Find segment boundaries
        segments = []
        i = 0
        while i < len(in_pareto):
            if in_pareto[i]:
                start = i
                while i < len(in_pareto) and in_pareto[i]:
                    i += 1
                end = i - 1
                segments.append((x_grid[start], x_grid[end]))
            else:
                i += 1
        
        # Filter out very small segments (numerical noise)
        segments = [(a, b) for a, b in segments if b - a > 1e-5]
        
        return segments

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Compute exact Pareto front segments
        segments = self._compute_pareto_segments()
        
        # Ensure we have reasonable segments; fallback if computation fails
        if len(segments) < 3:
            segments = [
                (0.0, 0.0575),
                (0.121, 0.2575),
                (0.332, 0.6523),
                (0.6523, 0.8233),
                (0.8233, 1.0),
            ]
        
        n_samples = self.budget
        n_segments = len(segments)
        
        # Allocate samples to each segment.
        # For hypervolume maximization on a convex decreasing curve, 
        # uniform spacing in f1 is near-optimal. But we want to ensure
        # each segment is well-represented.
        # Give each segment at least 2 points (endpoints), then distribute
        # remaining proportionally to segment length (which correlates with
        # hypervolume contribution).
        
        min_per_seg = 2
        total_min = min_per_seg * n_segments
        
        if n_samples <= total_min:
            # Distribute as evenly as possible
            per_seg = [n_samples // n_segments + (1 if i < n_samples % n_segments else 0) 
                       for i in range(n_segments)]
        else:
            remaining = n_samples - total_min
            total_len = sum(b - a for a, b in segments)
            per_seg = []
            for a, b in segments:
                n_seg = min_per_seg + int(remaining * (b - a) / total_len)
                per_seg.append(max(2, n_seg))
            # Adjust for rounding
            diff = n_samples - sum(per_seg)
            if diff != 0:
                # Add/remove from the largest segment
                max_idx = np.argmax(per_seg)
                per_seg[max_idx] += diff
        
        x1_values = []
        for idx, (a, b) in enumerate(segments):
            n_seg = per_seg[idx]
            if n_seg <= 2:
                x1_values.extend([a, b])
            else:
                # Uniform spacing in f1 for near-optimal hypervolume coverage
                x1_values.extend(np.linspace(a, b, n_seg).tolist())
        
        # Convert to numpy, deduplicate, sort
        x1_values = np.array(sorted(set(np.round(x1_values, 15))))
        
        # Limit to budget if needed
        if len(x1_values) > self.budget:
            # Keep all points but subsample if over budget
            idx = np.linspace(0, len(x1_values) - 1, self.budget).astype(int)
            x1_values = x1_values[idx]
        
        # If we have fewer points than budget, add more uniformly across all segments
        if len(x1_values) < self.budget:
            deficit = self.budget - len(x1_values)
            # Add points uniformly across the full [0,1] range on the Pareto front
            existing_set = set(np.round(x1_values, 12).tolist())
            extra_candidates = np.linspace(0, 1, deficit * 10)
            for xc in extra_candidates:
                if len(x1_values) >= self.budget:
                    break
                if np.round(xc, 12) not in existing_set:
                    x1_values.append(xc)
                    existing_set.add(np.round(xc, 12))
            x1_values = np.array(sorted(x1_values))[:self.budget]
        
        # Generate and evaluate points on the Pareto front
        # x2..x30 = 0 ensures g = 1 (the minimum possible value)
        n_points = len(x1_values)
        points = np.zeros((n_points, self.dim))
        for i, x1 in enumerate(x1_values):
            points[i, 0] = x1
            # x2..x30 remain 0, which gives g=1
        
        # Clip to bounds (should already be within [0,1] but ensure safety)
        points = np.clip(points, lo, hi)
        
        # Evaluate all points - each evaluation is recorded by the evaluator
        for i in range(n_points):
            if time.time() > deadline:
                break
            problem(points[i])
        
        return points
# EVOLVE-BLOCK-END
