"""Circle packing in perimeter-4 rectangle; maximize sum of radii for 21 circles."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
DEADLINE = 115.0  # seconds, safety margin before 120s limit

def _compute_radii(centers, w, h):
    """
    Compute the maximum radius for each circle such that they are disjoint and inside [0,w]x[0,h].
    This is a conservative estimate: r_i = min( dist_to_wall_i, min_j (dist_ij - r_j)/2 ) 
    is not quite right because r_j also depends on r_i.
    However, a simpler and robust approach for packing is:
    r_i = min( dist_to_wall_i, 0.5 * min_j dist_ij )
    This guarantees disjoint interiors if we use strict < or <= with epsilon.
    Actually, to maximize sum, we can solve the system:
    r_i + r_j <= dist_ij
    r_i <= dist_to_wall_i
    
    For speed and robustness in the optimizer, we can use a simple iterative relaxation or 
    just the "nearest neighbor" heuristic which is O(N^2) and very fast.
    Let's use the heuristic: 
    r_i = min( dist_to_wall_i, 0.5 * min_{j != i} dist_ij )
    This ensures no overlaps because dist_ij >= r_i + r_j is satisfied if r_i <= 0.5 dist_ij and r_j <= 0.5 dist_ij?
    No, if r_i = 0.5 d_ij and r_j = 0.5 d_jk, they might overlap with a third circle.
    
    Better approach: 
    Since N=21 is small, we can solve the LP quickly but with a safety margin.
    Or, we can use a simple iterative "push" method to find valid radii.
    
    Let's stick to the LP but add a safety factor to the constraints to avoid the overlap error.
    Or, simpler: Just use the "nearest neighbor" radius and verify.
    If we use r_i = min(dist_to_wall, 0.5 * min_dist_to_neighbor), it's valid.
    Is it optimal? No, but it's a good lower bound.
    
    Let's try a slightly better heuristic:
    1. Initialize r_i = dist_to_wall_i
    2. For each pair (i,j), if r_i + r_j > dist_ij, scale them down.
    
    Actually, the LP solution is the correct way to maximize sum for fixed centers.
    The issue was likely the tolerance in the evaluator or the LP solver.
    Let's use the LP but subtract a small epsilon from the radius in the final output.
    """
    n = len(centers)
    if n == 0:
        return np.array([])
    
    # Distance to walls
    wall_dists = np.minimum(np.minimum(centers[:, 0], w - centers[:, 0]), 
                            np.minimum(centers[:, 1], h - centers[:, 1]))
    
    # Distance between centers
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dists[i, j] = np.sqrt(dx*dx + dy*dy)
            
    # We want to find r such that r_i + r_j <= dists[i,j] and r_i <= wall_dists[i]
    # This is a LP. Let's solve it using a simple iterative method or scipy.
    # Given the previous failure, let's use a robust iterative solver.
    # Initialize r with wall distances
    r = wall_dists.copy()
    
    # Iterate to satisfy pairwise constraints
    for _ in range(20):
        for i in range(n):
            for j in range(i+1, n):
                d = dists[i, j]
                if r[i] + r[j] > d:
                    # Scale down both proportionally or just reduce the larger one?
                    # To maximize sum, we should reduce both such that r_i + r_j = d
                    # But this might violate other constraints.
                    # Simple fix: r_i = d - r_j, r_j = d - r_i? No.
                    # Let's just set r_i = max(0, d - r_j) and r_j = max(0, d - r_i)
                    # This is not optimal but ensures validity.
                    # Better: reduce both by half the excess.
                    excess = (r[i] + r[j] - d) / 2.0
                    r[i] -= excess
                    r[j] -= excess
                    # Ensure non-negative and within wall bounds
                    r[i] = np.clip(r[i], 0, wall_dists[i])
                    r[j] = np.clip(r[j], 0, wall_dists[j])
                    
    return r

def _evaluate_config(params, w, h):
    """
    Evaluate a configuration given parameters (centers flattened) and rectangle dimensions.
    Returns the sum of radii.
    """
    n = len(params) // 2
    centers = params.reshape(n, 2)
    # Clip centers to [0, w] x [0, h]
    centers[:, 0] = np.clip(centers[:, 0], 0, w)
    centers[:, 1] = np.clip(centers[:, 1], 0, h)
    
    radii = _compute_radii(centers, w, h)
    return np.sum(radii)

def construct():
    start_time = time.time()
    best_score = 0.0
    best_params = None
    best_w = 1.0
    best_h = 1.0
    best_circles = None

    np.random.seed(42)

    # Aspect ratios to try
    # Perimeter 4 => w+h=2.
    # Try squares and slight rectangles.
    aspect_ratios = [1.0, 1.1, 0.9, 1.2, 0.8, 1.5, 0.66]

    for ar in aspect_ratios:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break
            
        h = 2.0 / (ar + 1.0)
        w = 2.0 * ar / (ar + 1.0)
        
        # Initialize with a grid
        # 5x5 grid = 25 points, we need 21.
        # Or 4x6 = 24 points.
        # Let's use a jittered grid to break symmetry.
        rows = 4
        cols = 6
        grid_points = []
        for i in range(rows):
            for j in range(cols):
                x = (j + 0.5) * (w / cols)
                y = (i + 0.5) * (h / rows)
                # Add small jitter
                x += (np.random.rand() - 0.5) * 0.05 * w
                y += (np.random.rand() - 0.5) * 0.05 * h
                grid_points.append([x, y])
        
        # Select 21 points
        # Shuffle and take first 21? Or just take first 21?
        # Taking first 21 from a 4x6 grid leaves 3 empty spots.
        # It's better to randomize which 21 are kept or just use all 24 and drop 3 smallest?
        # The task requires exactly 21 circles.
        # Let's pick 21 random points from a larger candidate set to ensure diversity.
        # Or just use the first 21.
        
        candidates = grid_points[:N]
        params0 = np.array(candidates).flatten()
        
        # Optimize using Nelder-Mead
        def objective(params):
            elapsed = time.time() - start_time
            if elapsed > DEADLINE:
                return 0.0
            score = _evaluate_config(params, w, h)
            return -score

        # Run Nelder-Mead
        # To avoid getting stuck, we run a few restarts with different jitters
        n_restarts = 3
        if elapsed > DEADLINE - 30:
            n_restarts = 1

        for r in range(n_restarts):
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 5:
                break
                
            # Jitter the initial grid
            params0_jittered = params0.copy()
            for i in range(N):
                params0_jittered[2*i] += (np.random.rand() - 0.5) * 0.1 * w
                params0_jittered[2*i+1] += (np.random.rand() - 0.5) * 0.1 * h
            # Clip
            params0_jittered[0::2] = np.clip(params0_jittered[0::2], 0, w)
            params0_jittered[1::2] = np.clip(params0_jittered[1::2], 0, h)
            
            try:
                res = minimize(objective, params0_jittered, method='Nelder-Mead', 
                               options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
                
                if res.success or (time.time() - start_time) < DEADLINE:
                    current_score = -res.fun
                    if current_score > best_score:
                        best_score = current_score
                        best_params = res.x.copy()
                        best_w = w
                        best_h = h
            except Exception:
                pass

    # If no good result, fallback to grid
    if best_params is None:
        w = 1.0
        h = 1.0
        centers = np.zeros((N, 2))
        # Simple 5x5 grid, take 21
        idx = 0
        for i in range(5):
            for j in range(5):
                if idx < N:
                    centers[idx] = [j + 0.5, i + 0.5]
                    idx += 1
        best_params = centers.flatten()
        best_w = w
        best_h = h

    # Final calculation of circles
    centers = best_params.reshape(N, 2)
    centers[:, 0] = np.clip(centers[:, 0], 0, best_w)
    centers[:, 1] = np.clip(centers[:, 1], 0, best_h)
    
    radii = _compute_radii(centers, best_w, best_h)
    
    # Apply a small safety factor to radii to ensure strict disjointness
    # The evaluator checks dist < r_i + r_j - 1e-9.
    # So if dist == r_i + r_j, it fails.
    # We must ensure dist >= r_i + r_j + epsilon.
    # Let's scale radii by 0.999 or subtract a small value.
    # A safer approach: check and reduce if necessary.
    
    circles = np.column_stack([centers, radii])
    
    # Validation and repair
    for i in range(N):
        for j in range(i + 1, N):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            sum_r = circles[i, 2] + circles[j, 2]
            if dist < sum_r + 1e-9:
                # Reduce radii to fix overlap
                # We need dist >= sum_r_new + 1e-9
                # Let's reduce both by half the excess
                excess = (sum_r - dist + 1e-9) / 2.0
                circles[i, 2] -= excess
                circles[j, 2] -= excess
                circles[i, 2] = max(circles[i, 2], 0.0)
                circles[j, 2] = max(circles[j, 2], 0.0)
                
    # Check wall constraints
    for i in range(N):
        min_x = circles[i, 0] - circles[i, 2]
        max_x = circles[i, 0] + circles[i, 2]
        min_y = circles[i, 1] - circles[i, 2]
        max_y = circles[i, 1] + circles[i, 2]
        
        if min_x < 0:
            circles[i, 2] = min(circles[i, 2], circles[i, 0])
        if max_x > best_w:
            circles[i, 2] = min(circles[i, 2], best_w - circles[i, 0])
        if min_y < 0:
            circles[i, 2] = min(circles[i, 2], circles[i, 1])
        if max_y > best_h:
            circles[i, 2] = min(circles[i, 2], best_h - circles[i, 1])
            
        circles[i, 2] = max(circles[i, 2], 0.0)

    return circles
