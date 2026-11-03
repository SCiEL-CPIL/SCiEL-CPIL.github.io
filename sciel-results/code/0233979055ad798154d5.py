"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    best_score = -1
    best_result = None
    
    # Perimeter 4 => W + H = 2
    # We will try a range of aspect ratios.
    # Focus on near-square and slightly rectangular to accommodate hexagonal packing.
    
    aspect_ratios = [1.0, 0.95, 1.05, 0.9, 1.1, 0.85, 1.15, 0.8, 1.2]
    
    # Grid resolution: 
    # Too fine -> too many candidates, slow local search.
    # Too coarse -> poor resolution of optimal positions.
    # Let's use a grid of 50x50 points for the rectangle.
    # Candidates: 2500 points.
    # Local search on 21 items from 2500 candidates.
    
    # We will use a "Greedy + Local Search" approach.
    # 1. Generate candidate points on a grid.
    # 2. For each point, the max possible radius is determined by distance to walls and other circles.
    #    However, radii are coupled. 
    #    Alternative: Fixed radius search? No, we maximize sum.
    #    
    #    Better Discrete Strategy:
    #    Assume a uniform radius R? No, optimal might have varying radii.
    #    But often, in dense packings, radii are similar.
    #    Let's try to optimize a single radius R and positions?
    #    If we fix R, we are looking for max number of circles? No, we need exactly 21.
    #    If we fix R, we check if 21 circles of radius R fit.
    #    We can binary search on R? 
    #    But the objective is sum of radii. If we find max uniform R, sum = 21*R.
    #    Is uniform radius optimal? Not necessarily, but a good baseline.
    #    
    #    Let's try a hybrid:
    #    1. Find a good uniform radius packing using grid search / local search.
    #    2. Then, allow radii to vary slightly? Or just stick to uniform for speed and robustness?
    #    The inspiration suggests "max-weight independent set".
    #    If we assume uniform radius r, the "weight" is just 1 for each circle.
    #    We want to place 21 circles of radius r such that they don't overlap.
    #    We can binary search for the maximum r such that 21 circles fit.
    #    
    #    Algorithm:
    #    1. For a given aspect ratio (w, h):
    #       a. Define a grid of potential centers.
    #       b. Binary search on radius r.
    #       c. For a given r, determine which grid points are valid (distance to wall >= r).
    #       d. Use a greedy algorithm or local search to place 21 circles of radius r.
    #          - Greedy: Sort candidates by some heuristic (e.g., distance to center, or "tightness"), pick one, remove neighbors within 2r.
    #          - Local Search: Start with a greedy solution, try to swap one circle for another to allow a larger r? 
    #          - Actually, for fixed r, we just need to check if a packing of size 21 exists.
    #          - We can use a simple backtracking or greedy with randomization.
    #    
    #    Let's refine:
    #    For each aspect ratio:
    #       1. Determine max possible r (min(w,h)/2).
    #       2. Binary search r in [0, min(w,h)/2].
    #          - For each r, run a "Packing Feasibility Check".
    #          - Feasibility Check:
    #             - Generate all valid grid points (cx, cy) where cx>=r, cx<=w-r, cy>=r, cy<=h-r.
    #             - We need to select 21 points such that dist(p_i, p_j) >= 2r.
    #             - This is a Maximum Independent Set problem on a graph where nodes are valid points and edges connect points with dist < 2r.
    #             - MIS is NP-hard, but for 21 nodes and grid structure, we can use a heuristic.
    #             - Heuristic: 
    #                 1. Randomly shuffle valid points.
    #                 2. Greedily select points if they don't conflict with already selected.
    #                 3. If we get 21, feasible.
    #                 4. Repeat several times with different shuffles to increase chance of finding a packing.
    #                 5. If found, return True.
    #    
    #    This binary search will find the max uniform radius r_uniform.
    #    Score = 21 * r_uniform.
    #    
    #    Can we do better with non-uniform radii?
    #    Maybe. But uniform is a strong baseline. 
    #    Let's also try a "Variable Radius" local search after finding a uniform packing.
    #    Start with centers from uniform packing.
    #    Initialize radii to r_uniform.
    #    Perform local moves: 
    #       - Move one center slightly.
    #       - Adjust radii to fill available space? 
    #    This is getting complex. Let's stick to the Discrete Uniform Radius approach first, 
    #    as it directly addresses the "discrete combinatorial" directive and avoids continuous optimization pitfalls.
    
    grid_size = 60  # 60x60 grid
    
    def check_feasibility(w, h, r, n_circles, n_trials=50):
        """
        Checks if n_circles of radius r can be packed in w x h rectangle
        using a grid of candidate centers.
        Returns True if a packing is found.
        """
        if r <= 0:
            return True
        if 2*r > w or 2*r > h:
            return False
        
        # Generate valid grid points
        # x in [r, w-r], y in [r, h-r]
        # Step size: small enough to allow flexibility, but not too many points.
        # Let's use the same grid resolution for all r? 
        # The validity of a point depends on r.
        
        # To speed up, we can pre-compute a grid of points and filter.
        # But the "neighbors" change with r.
        
        # Let's generate points on the fly for the valid region.
        # Number of points: (w-2r)/step * (h-2r)/step.
        # If step is too small, too many points.
        # Let's fix the number of grid points in the valid region to be around 100-200.
        
        # Actually, a fixed spatial grid is better for consistency.
        # Let's use a fixed grid of 60x60 over the [0,w]x[0,h] area.
        
        xs = np.linspace(r, w - r, grid_size)
        ys = np.linspace(r, h - r, grid_size)
        
        # Create meshgrid
        XX, YY = np.meshgrid(xs, ys)
        points = np.column_stack((XX.ravel(), YY.ravel()))
        
        # Filter points that are actually valid (they are by construction of xs, ys)
        # But we must ensure they are within bounds.
        
        # We need to select n_circles points such that pairwise distance >= 2r.
        
        # Greedy with Randomization
        for _ in range(n_trials):
            # Shuffle points
            idx = np.random.permutation(len(points))
            shuffled_points = points[idx]
            
            selected = []
            for p in shuffled_points:
                # Check against already selected
                valid = True
                for s in selected:
                    dx = p[0] - s[0]
                    dy = p[1] - s[1]
                    if dx*dx + dy*dy < (2*r)**2 - 1e-9:
                        valid = False
                        break
                if valid:
                    selected.append(p)
                    if len(selected) == n_circles:
                        return True
            
            if len(selected) == n_circles:
                return True
                
        return False

    def find_max_uniform_radius(w, h, n_circles, max_iter=30):
        """
        Binary search for the maximum uniform radius.
        """
        lo = 0.0
        hi = min(w, h) / 2.0
        
        # Quick check if hi is feasible
        if not check_feasibility(w, h, hi, n_circles, n_trials=5):
            # Binary search
            for _ in range(max_iter):
                if time.time() > deadline - 5:
                    break
                mid = (lo + hi) / 2.0
                if check_feasibility(w, h, mid, n_circles, n_trials=20):
                    lo = mid
                else:
                    hi = mid
        else:
            lo = hi
            
        return lo

    for w_ratio in aspect_ratios:
        if time.time() > deadline:
            break
            
        w = 2.0 * w_ratio / (1.0 + w_ratio)
        h = 2.0 - w
        
        # Ensure w, h > 0
        if w <= 0 or h <= 0:
            continue
            
        # Find max uniform radius
        r_max = find_max_uniform_radius(w, h, N)
        
        if r_max > 0:
            # Now, recover the positions for this r_max
            # We need to run the feasibility check one more time to get the positions
            # Modify check_feasibility to return positions?
            
            # Let's write a specific function to get positions
            def get_positions(w, h, r, n_circles, n_trials=100):
                xs = np.linspace(r, w - r, grid_size)
                ys = np.linspace(r, h - r, grid_size)
                XX, YY = np.meshgrid(xs, ys)
                points = np.column_stack((XX.ravel(), YY.ravel()))
                
                best_sel = None
                for _ in range(n_trials):
                    idx = np.random.permutation(len(points))
                    shuffled_points = points[idx]
                    selected = []
                    for p in shuffled_points:
                        valid = True
                        for s in selected:
                            dx = p[0] - s[0]
                            dy = p[1] - s[1]
                            if dx*dx + dy*dy < (2*r)**2 - 1e-9:
                                valid = False
                                break
                        if valid:
                            selected.append(p)
                            if len(selected) == n_circles:
                                return np.array(selected)
                    if len(selected) == n_circles:
                        if best_sel is None or len(selected) > len(best_sel):
                            best_sel = np.array(selected)
                return best_sel

            positions = get_positions(w, h, r_max, N)
            
            if positions is not None and len(positions) == N:
                # Construct result array
                # Centers: positions[:, 0], positions[:, 1]
                # Radii: all r_max
                result = np.zeros((N, 3))
                result[:, 0] = positions[:, 0]
                result[:, 1] = positions[:, 1]
                result[:, 2] = r_max
                
                # Validate
                valid = True
                for i in range(N):
                    cx, cy, r = result[i]
                    if cx < r - 1e-7 or cx > w - r + 1e-7 or cy < r - 1e-7 or cy > h - r + 1e-7:
                        valid = False
                        break
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            dx = result[i, 0] - result[j, 0]
                            dy = result[i, 1] - result[j, 1]
                            dist = math.hypot(dx, dy)
                            if dist < result[i, 2] + result[j, 2] - 1e-7:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    score = np.sum(result[:, 2])
                    if score > best_score:
                        best_score = score
                        best_result = result

    # Fallback: simple grid if optimization failed
    if best_result is None:
        # 5x5 grid in 1x1 square
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append((cx, cy, r))
        best_result = np.array(cells[:N])
        
    return best_result
# EVOLVE-BLOCK-END
