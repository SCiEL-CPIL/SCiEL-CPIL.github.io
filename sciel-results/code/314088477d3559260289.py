# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    def calc_radius(pts):
        """Calculate the maximum possible radius for a given set of points."""
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dist_mat, np.inf)
        min_pair_dist = np.min(dist_mat)
        
        # Distance to boundaries
        d_b = np.min([
            0.5 - pts[:, 0],
            0.5 + pts[:, 0],
            0.5 - pts[:, 1],
            0.5 + pts[:, 1]
        ], axis=0)
        min_boundary_dist = np.min(d_b)
        
        return min(min_pair_dist / 2.0, min_boundary_dist)

    def init_hex_grid(n_points):
        """Initialize points in a hexagonal grid pattern."""
        pts_list = []
        s = 0.05
        # Generate a grid larger than the unit square
        for i in range(-12, 13):
            for j in range(-12, 13):
                x = i * s + (j % 2) * s * 0.5
                y = j * s * np.sqrt(3) / 2
                pts_list.append([x, y])
        
        pts_arr = np.array(pts_list)
        
        # Filter to those inside [-0.5, 0.5]
        mask = (pts_arr[:, 0] >= -0.5) & (pts_arr[:, 0] <= 0.5) & (pts_arr[:, 1] >= -0.5) & (pts_arr[:, 1] <= 0.5)
        pts_arr = pts_arr[mask]
        
        if len(pts_arr) >= n_points:
            indices = np.random.choice(len(pts_arr), n_points, replace=False)
            pts_arr = pts_arr[indices]
        else:
            # Pad with random points if not enough
            diff = n_points - len(pts_arr)
            random_pts = (np.random.rand(diff, 2) - 0.5) * 0.9
            pts_arr = np.vstack([pts_arr, random_pts])
            
        return pts_arr

    def push_optimize(points, time_limit):
        """Optimize using discrete push operations on the tightest constraint."""
        points = points.copy()
        best_points = points.copy()
        best_r = calc_radius(points)
        
        # Small step size
        step = 0.002
        stagnation_count = 0
        max_stagnation = 500
        
        while time.time() < time_limit:
            # 1. Calculate current radius and identify bottleneck
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist_mat = np.sqrt(np.sum(diff**2, axis=-1))
            np.fill_diagonal(dist_mat, np.inf)
            
            # Find min pair distance
            min_pair_dist = np.min(dist_mat)
            # Find indices of min pair
            idx_flat = np.argmin(dist_mat)
            i, j = np.unravel_index(idx_flat, dist_mat.shape)
            
            # Find min wall distance
            d_x_neg = points[:, 0] + 0.5
            d_x_pos = 0.5 - points[:, 0]
            d_y_neg = points[:, 1] + 0.5
            d_y_pos = 0.5 - points[:, 1]
            
            wall_dists = np.column_stack([d_x_neg, d_x_pos, d_y_neg, d_y_pos])
            min_wall_dist = np.min(wall_dists)
            idx_wall_flat = np.argmin(wall_dists)
            k, wall_idx = np.unravel_index(idx_wall_flat, wall_dists.shape)
            
            # Determine which constraint is tighter
            r_pair = min_pair_dist / 2.0
            r_wall = min_wall_dist
            
            if r_pair < r_wall:
                # Push pair i, j apart
                vec = points[j] - points[i]
                norm = np.linalg.norm(vec)
                if norm > 1e-9:
                    unit_vec = vec / norm
                    points[i] -= unit_vec * step
                    points[j] += unit_vec * step
            else:
                # Push point k away from wall
                if wall_idx == 0: # x < -0.5
                    points[k, 0] += step
                elif wall_idx == 1: # x > 0.5
                    points[k, 0] -= step
                elif wall_idx == 2: # y < -0.5
                    points[k, 1] += step
                else: # y > 0.5
                    points[k, 1] -= step
            
            # Clamp to bounds
            points = np.clip(points, -0.5, 0.5)
            
            # Check improvement
            current_r = calc_radius(points)
            if current_r > best_r:
                best_r = current_r
                best_points = points.copy()
                stagnation_count = 0
            else:
                stagnation_count += 1
                if stagnation_count > max_stagnation:
                    # Add small random perturbation to escape local minimum
                    points += (np.random.rand(N, 2) - 0.5) * 0.005
                    points = np.clip(points, -0.5, 0.5)
                    stagnation_count = 0
        
        return best_points, best_r

    # 1. Initialization
    np.random.seed(42)
    init_pts = init_hex_grid(N)
    
    # 2. Optimize
    # Give it most of the time
    opt_pts, r = push_optimize(init_pts, deadline - 5.0)
    
    best_points = opt_pts
    best_r = r
    
    # 3. If time remains, try a random restart
    if time.time() < deadline - 10.0:
        random_pts = (np.random.rand(N, 2) - 0.5)
        # Ensure they are inside
        random_pts = np.clip(random_pts, -0.49, 0.49)
        
        opt_pts2, r2 = push_optimize(random_pts, deadline - 2.0)
        
        if r2 > best_r:
            best_r = r2
            best_points = opt_pts2
            
    return best_points
# EVOLVE-BLOCK-END
