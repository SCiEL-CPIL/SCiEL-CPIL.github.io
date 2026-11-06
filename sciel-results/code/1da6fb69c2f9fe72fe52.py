"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

def _compute_radius(centers):
    """Compute the maximum valid radius for the given centers."""
    if centers is None or len(centers) == 0:
        return 0.0
    
    # Check for non-finite values
    if not np.all(np.isfinite(centers)):
        return 0.0
        
    # Check if centers are within the square [-0.5, 0.5]
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0

    # 1. Distance to walls
    min_x = np.min(centers[:, 0])
    max_x = np.max(centers[:, 0])
    min_y = np.min(centers[:, 1])
    max_y = np.max(centers[:, 1])
    
    r_wall = min(0.5 + min_x, 0.5 - max_x, 0.5 + min_y, 0.5 - max_y)
    
    # 2. Distance between centers
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    
    min_pair_dist = np.min(dist)
    r_pair = min_pair_dist / 2.0
    
    return min(r_wall, r_pair)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Strategy: Discrete Max-Min Facility Location
    # 1. Define a high-resolution grid
    # 2. Iteratively place points to maximize the minimum distance to existing points and walls
    # 3. Refine using local continuous optimization
    
    # Grid resolution
    # r ~ 0.036, so 1/r ~ 27. Grid spacing should be smaller than r.
    # Let's use a grid of 100x100 points. Spacing = 1/99 ~ 0.01.
    # This allows precise placement.
    grid_size = 100
    x_coords = np.linspace(-0.5, 0.5, grid_size)
    y_coords = np.linspace(-0.5, 0.5, grid_size)
    
    # Create meshgrid
    X, Y = np.meshgrid(x_coords, y_coords)
    grid_points = np.column_stack((X.ravel(), Y.ravel()))
    
    # Precompute distance to walls for each grid point
    # d_wall = min(x + 0.5, 0.5 - x, y + 0.5, 0.5 - y)
    d_walls = np.min([
        grid_points[:, 0] + 0.5,
        0.5 - grid_points[:, 0],
        grid_points[:, 1] + 0.5,
        0.5 - grid_points[:, 1]
    ], axis=0)
    
    # Current set of centers
    centers = []
    # We will store indices of selected grid points for fast distance update?
    # Or just keep the list of coordinates.
    
    # For efficiency, we can maintain a "score" array for each grid point:
    # score[i] = min(d_wall[i], min_{j in centers} dist(grid_point[i], center[j]))
    # We want to pick the grid point with the max score.
    # When we add a new center, we update scores:
    # score[i] = min(score[i], dist(grid_point[i], new_center))
    
    scores = d_walls.copy()
    
    # Optimization: We only care about points with score > current_best_score.
    # But updating scores for all 10,000 points 200 times is 2,000,000 distance calculations.
    # Each calculation is vectorized over 10,000 points.
    # 10,000 * 200 = 2M operations. This is very fast in NumPy (<0.1s).
    
    for k in range(N):
        if time.time() > deadline:
            break
            
        # Find the grid point with the maximum score
        max_idx = np.argmax(scores)
        new_center = grid_points[max_idx]
        centers.append(new_center)
        
        # Update scores with the new center
        # dist = ||grid_points - new_center||
        diff = grid_points - new_center
        dists = np.sqrt(np.sum(diff**2, axis=1))
        scores = np.minimum(scores, dists)
        
    centers = np.array(centers)
    
    # The discrete solution gives a lower bound on the radius.
    # Now refine using local continuous optimization.
    # We use a "push-apart" algorithm but initialized with the discrete solution.
    # This should converge faster and potentially to a better local minimum.
    
    # Compute initial radius
    current_r = _compute_radius(centers)
    
    # Local optimization loop
    # We use a simple iterative improvement:
    # 1. Find the bottleneck (min pair or wall)
    # 2. Try to move points to increase the bottleneck
    
    # To make it more robust, we can do a few rounds of "random perturbation + local push"
    # But let's stick to a deterministic local search first for stability.
    
    # Better approach: Simulated Annealing or Gradient Descent on the objective function.
    # Objective: max min(distances). This is non-smooth.
    # Let's use the "bottleneck push" strategy from the seed but with a better init.
    
    # We will run a multi-stage local refinement.
    
    # Stage 1: Fine-tune positions using gradient-like steps on the bottleneck.
    # Since the discrete grid is fine (0.01), the centers are already close to optimal.
    # We can try small random perturbations and accept if radius increases.
    
    best_centers = centers.copy()
    best_r = current_r
    
    # Local search with random restarts/perturbations
    # We have ~100s left.
    # Each evaluation of radius takes O(N^2) = 40,000 ops. Very fast.
    # We can do many iterations.
    
    # Let's do a simple hill climbing with random moves.
    # Move one point at a time randomly within a small radius, accept if radius improves.
    
    # To be more effective, we can identify "free" points (those not in bottleneck) and move them?
    # Or just random.
    
    # Actually, a more structured approach:
    # Use scipy.optimize to maximize the radius?
    # Maximize r subject to:
    #   ||c_i - c_j|| >= 2r for all i<j
    #   c_ik + r <= 0.5, c_ik - r >= -0.5
    
    # This is a Non-Linear Programming (NLP) problem.
    # We can use scipy.optimize.minimize on -r.
    # Variables: c_1...c_N, r. Total 401 variables.
    # Constraints: 19900 pairwise + 800 wall constraints.
    # This might be slow or get stuck.
    
    # Let's stick to the heuristic push-apart but with a tighter loop and better init.
    
    # Refine loop
    step_size = 0.005
    for i in range(5000):
        if time.time() > deadline:
            break
            
        # Compute current radius and bottleneck
        # To save time, we can compute distances once.
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        min_pair_dist = np.min(dist)
        r_pair = min_pair_dist / 2.0
        
        min_x = np.min(centers[:, 0])
        max_x = np.max(centers[:, 0])
        min_y = np.min(centers[:, 1])
        max_y = np.max(centers[:, 1])
        
        r_wall = min(0.5 + min_x, 0.5 - max_x, 0.5 + min_y, 0.5 - max_y)
        
        r_current = min(r_pair, r_wall)
        
        if r_current > best_r:
            best_r = r_current
            best_centers = centers.copy()
            
        # Determine bottleneck
        if r_pair < r_wall:
            # Pair bottleneck
            idx = np.argmin(dist)
            i_idx, j_idx = np.unravel_index(idx, dist.shape)
            vec = centers[i_idx] - centers[j_idx]
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                unit_vec = vec / norm
                # Push apart
                centers[i_idx] += unit_vec * step_size
                centers[j_idx] -= unit_vec * step_size
        else:
            # Wall bottleneck
            # Find which wall
            d_left = 0.5 + min_x
            d_right = 0.5 - max_x
            d_bottom = 0.5 + min_y
            d_top = 0.5 - max_y
            
            min_d = min(d_left, d_right, d_bottom, d_top)
            
            if min_d == d_left:
                idx = np.argmin(centers[:, 0])
                centers[idx, 0] += step_size
            elif min_d == d_right:
                idx = np.argmax(centers[:, 0])
                centers[idx, 0] -= step_size
            elif min_d == d_bottom:
                idx = np.argmin(centers[:, 1])
                centers[idx, 1] += step_size
            else:
                idx = np.argmax(centers[:, 1])
                centers[idx, 1] -= step_size
        
        # Clamp
        centers = np.clip(centers, -0.5, 0.5)
        
        # Decay step size
        step_size *= 0.999
        
    # Final check
    final_r = _compute_radius(best_centers)
    if final_r > _compute_radius(centers):
        return best_centers
    else:
        return centers

# EVOLVE-BLOCK-END
