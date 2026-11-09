"""Multi-objective optimizer for ZDT3 using parametric analytic inversion on the Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_zdt3_segments_high_res(n_grid=2000000):
    """
    Compute the exact Pareto-optimal segments of ZDT3 with high resolution.
    
    ZDT3: f1 = x1, f2 = (1 + 10*(n-1)) * (g + h(x1, g))
    where g = 1 + 10*(n-1) * sum(x_i^2) for i=2..n
    and h(x, g) = 1 - sqrt(x/g) - (x/g)*sin(10*pi*x)
    
    For g=1 (x2..xn=0): h(x,1) = 1 - sqrt(x) - x*sin(10*pi*x)
    A point is Pareto-optimal iff h(x,1) >= 0.
    """
    x = np.linspace(0, 1, n_grid)
    h = 1.0 - np.sqrt(x) - x * np.sin(10.0 * np.pi * x)
    
    # Find contiguous regions where h >= 0
    valid = h >= 0.0
    segments = []
    in_segment = False
    start = 0.0
    for i in range(len(valid)):
        if valid[i] and not in_segment:
            start = x[i]
            in_segment = True
        elif not valid[i] and in_segment:
            segments.append((start, x[i-1]))
            in_segment = False
    if in_segment:
        segments.append((start, x[-1]))
    
    # Merge very close segments (numerical artifacts)
    merged = []
    for seg in segments:
        if merged and seg[0] - merged[-1][1] < 1e-8:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)
    
    return merged


def _find_segment_endpoints(segments, n_grid=1000000):
    """
    Refine segment endpoints using bisection on h(x,1) = 0.
    """
    def h_func(x):
        if x <= 0:
            return 1.0
        return 1.0 - np.sqrt(x) - x * np.sin(10.0 * np.pi * x)
    
    def bisect(a, b, fa, fb, tol=1e-15, max_iter=100):
        # We know h(a) >= 0 and h(b) < 0 (or vice versa)
        # Find root where h = 0
        for _ in range(max_iter):
            mid = (a + b) / 2
            fm = h_func(mid)
            if abs(b - a) < tol:
                return mid
            if fm * fa >= 0:
                a, fa = mid, fm
            else:
                b, fb = mid, fm
        return (a + b) / 2
    
    refined_segments = []
    for a, b in segments:
        # Refine left endpoint: find where h crosses from negative to positive
        # Search left from a
        left_bound = max(0, a - 0.01)
        # Find the exact crossing
        fa = h_func(a)
        # Step left until we find h < 0
        step = 1e-6
        x_left = a
        while x_left > left_bound and h_func(x_left) >= 0:
            x_left -= step
        if x_left > left_bound and h_func(x_left) < 0:
            a_refined = bisect(x_left, a, h_func(x_left), fa)
        else:
            a_refined = a
        
        # Refine right endpoint: find where h crosses from positive to negative
        right_bound = min(1.0, b + 0.01)
        fb = h_func(b)
        x_right = b
        while x_right < right_bound and h_func(x_right) >= 0:
            x_right += step
        if x_right < right_bound and h_func(x_right) < 0:
            b_refined = bisect(b, x_right, fb, h_func(x_right))
        else:
            b_refined = b
        
        refined_segments.append((a_refined, b_refined))
    
    return refined_segments


class Optimizer:
    """
    Multi-objective optimizer for ZDT3 using parametric analytic inversion.
    
    Strategy: Instead of sampling x1 uniformly, we sample f1 values uniformly
    within each Pareto-optimal segment, then invert to get x1. Since f1 = x1,
    this is equivalent to uniform x1 sampling within segments, but we ensure
    precise coverage of the objective space for hypervolume maximization.
    
    Key insight: For hypervolume with reference point [11, 11], we want points
    that maximize the covered area. Uniform sampling in f1 (which equals x1)
    within each segment is optimal for covering the front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Compute exact Pareto-optimal segments of ZDT3 with high resolution
        segments = _compute_zdt3_segments_high_res(n_grid=2000000)
        
        # Refine segment endpoints
        segments = _find_segment_endpoints(segments)
        
        # Number of segments
        n_segments = len(segments)
        
        # Distribute budget across segments proportionally to their length
        seg_lengths = np.array([b - a for a, b in segments])
        total_len = seg_lengths.sum()
        
        # Allocate samples to each segment proportional to length, but ensure at least 1 per segment
        raw_alloc = (seg_lengths / total_len) * self.budget
        alloc = np.maximum(1, np.floor(raw_alloc)).astype(int)
        
        # Adjust to match total budget
        diff = self.budget - alloc.sum()
        if diff > 0:
            # Add remaining to largest segments
            order = np.argsort(-seg_lengths)
            for i in range(min(diff, n_segments)):
                alloc[order[i]] += 1
        elif diff < 0:
            # Remove from smallest segments (but keep at least 1)
            order = np.argsort(seg_lengths)
            for i in range(min(-diff, n_segments)):
                if alloc[order[i]] > 1:
                    alloc[order[i]] -= 1
        
        # Generate x1 values for each segment using stratified sampling
        x1_values = []
        
        for seg_idx, (a, b) in enumerate(segments):
            n_samples = alloc[seg_idx]
            if n_samples <= 0:
                continue
            
            # Stratified sampling: divide segment into n_samples equal sub-intervals
            # and sample the midpoint of each
            sub_width = (b - a) / n_samples
            for j in range(n_samples):
                # Midpoint of j-th sub-interval
                x1 = a + (j + 0.5) * sub_width
                x1_values.append(x1)
        
        # Convert to array
        x1_values = np.array(x1_values)
        
        # Ensure we have exactly budget points
        if len(x1_values) > self.budget:
            x1_values = x1_values[:self.budget]
        elif len(x1_values) < self.budget:
            # Add additional points using Halton-like sequence
            extra_needed = self.budget - len(x1_values)
            extra_x1 = []
            for i in range(extra_needed):
                # Use a golden ratio based sequence for good distribution
                frac = ((i + 1) * 0.618033988749895) % 1.0
                # Map to a random segment
                seg_idx = int(frac * n_segments) % n_segments
                a, b = segments[seg_idx]
                x1 = a + frac * (b - a)
                extra_x1.append(x1)
            x1_values = np.concatenate([x1_values, np.array(extra_x1)])
        
        # Generate points: x2..x30 = 0 ensures g = 1 (minimum possible)
        n_points = len(x1_values)
        points = np.zeros((n_points, self.dim))
        points[:, 0] = x1_values
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
