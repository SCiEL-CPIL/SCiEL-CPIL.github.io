"""Black-box continuous optimizer on the BBOB-style function f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq


class Optimizer:
    """Adaptive Lattice Refinement optimizer for deterministic systematic search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275.0  # safety margin before 300s limit
        d = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = 0

        def wrapped_func(x):
            nonlocal call_count
            if call_count >= budget or time.time() > deadline:
                return np.inf
            call_count += 1
            try:
                fx = func(x)
                return fx
            except Exception:
                return np.inf

        # Phase 1: Coarse grid initialization
        # Evaluate a coarse grid to identify promising regions
        coarse_steps = 5  # 5 points per dimension -> 5^5 = 3125 points (too many)
        # Use fewer: 3 per dim -> 3^5 = 243 points
        coarse_steps = 3
        grid_points = []
        for i in range(coarse_steps):
            for j in range(coarse_steps):
                for k in range(coarse_steps):
                    for l in range(coarse_steps):
                        for m in range(coarse_steps):
                            x = np.array([-4 + 8 * i / (coarse_steps - 1),
                                          -4 + 8 * j / (coarse_steps - 1),
                                          -4 + 8 * k / (coarse_steps - 1),
                                          -4 + 8 * l / (coarse_steps - 1),
                                          -4 + 8 * m / (coarse_steps - 1)])
                            grid_points.append(x)

        # Evaluate coarse grid (limit to budget)
        coarse_budget = min(len(grid_points), budget // 4)
        for x in grid_points[:coarse_budget]:
            if call_count >= budget or time.time() > deadline:
                break
            fx = wrapped_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            # Fallback
            best_x = np.zeros(d)
            best_f = wrapped_func(best_x)

        # Phase 2: Adaptive Lattice Refinement
        # Maintain a priority queue of regions (hyper-rectangles) to subdivide
        # Each region: (priority, center, lower_bounds, upper_bounds, depth)
        # Priority = function value at center (we want to explore low-value regions)
        
        # Initialize with regions around the best point found
        # Create initial regions: split the space around best_x
        region_size = 2.0  # Initial region half-width
        regions = []
        
        # Create a few initial regions centered near best_x and spread out
        offsets = [
            np.zeros(d),
            np.ones(d) * region_size,
            -np.ones(d) * region_size,
            np.array([region_size, 0, 0, 0, 0]),
            np.array([0, region_size, 0, 0, 0]),
            np.array([0, 0, region_size, 0, 0]),
            np.array([0, 0, 0, region_size, 0]),
            np.array([0, 0, 0, 0, region_size]),
        ]
        
        for offset in offsets:
            center = best_x + offset
            center = np.clip(center, -5, 5)
            lower = center - region_size
            upper = center + region_size
            lower = np.clip(lower, -5, 5)
            upper = np.clip(upper, -5, 5)
            
            # Evaluate center
            fx = wrapped_func(center)
            if fx < best_f:
                best_f = fx
                best_x = center.copy()
            
            # Add to heap (min-heap: lowest function value first)
            heapq.heappush(regions, (fx, center, lower, upper, 0))

        # Refinement loop
        max_depth = 6  # Maximum subdivision depth
        min_region_size = 0.1  # Stop subdividing if region is too small
        
        while call_count < budget and time.time() < deadline:
            if not regions:
                break
            
            # Get the most promising region (lowest center value)
            priority, center, lower, upper, depth = heapq.heappop(regions)
            
            # Check if region is too small to subdivide
            if depth >= max_depth:
                continue
            
            region_width = upper - lower
            if np.min(region_width) < min_region_size:
                continue
            
            # Subdivide this region into 2^d sub-regions
            # For efficiency, subdivide along the largest dimension first
            # Create sub-regions by splitting each dimension in half
            sub_regions = []
            for mask in range(2 ** d):
                new_center = np.zeros(d)
                new_lower = np.zeros(d)
                new_upper = np.zeros(d)
                
                for i in range(d):
                    mid = (lower[i] + upper[i]) / 2
                    if (mask >> i) & 1:
                        new_lower[i] = mid
                        new_upper[i] = upper[i]
                    else:
                        new_lower[i] = lower[i]
                        new_upper[i] = mid
                    new_center[i] = (new_lower[i] + new_upper[i]) / 2
                
                # Evaluate center of sub-region
                fx = wrapped_func(new_center)
                if fx < best_f:
                    best_f = fx
                    best_x = new_center.copy()
                
                sub_regions.append((fx, new_center, new_lower, new_upper, depth + 1))
            
            # Add sub-regions to heap
            for region in sub_regions:
                heapq.heappush(regions, region)

            # Periodically check time
            if time.time() > deadline:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
