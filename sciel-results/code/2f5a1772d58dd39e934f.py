"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    # 1. Initialization: Jittered grid
    # This provides a good initial spread.
    m = int(np.ceil(np.sqrt(N)))
    # We need exactly N points.
    # Create a grid of m x m points.
    # If m*m > N, we will remove some.
    # If m*m < N, we need to increase m.
    
    # Let's use a grid that is slightly larger than needed to ensure coverage, then pick N points.
    # Or just use a rectangular grid that fits N.
    # 14x14 = 196. 15x15 = 225.
    # Let's use 15x15 and randomly select 200? No, that breaks structure.
    # Let's use a 15x15 grid and remove the last 25 in a deterministic way (e.g. corners or center).
    # Or simpler: Use a 14x15 grid (210 points) and remove 10.
    
    # Let's stick to the jittered grid approach but ensure it's robust.
    m = 15
    grid_points = []
    for i in range(m):
        for j in range(m):
            x = (i + 0.5) / m - 0.5
            y = (j + 0.5) / m - 0.5
            grid_points.append([x, y])
    
    # We have 225 points. We need 200.
    # Shuffle and take first 200? This might create large gaps.
    # Better: Take all points, but if we have too many, remove those closest to the center?
    # Or just use the first 200 in a spiral order?
    # Let's use a simple random selection from the grid, but with a fixed seed for reproducibility?
    # The task allows random, but deterministic is better for evaluation consistency if possible.
    # However, the evaluator runs multiple times? "The evaluator measures the raw value".
    # It says "Each evaluation must finish within 120 s".
    # It doesn't say deterministic.
    # Let's use a fixed seed for initialization to be safe, or just use the first N points of a well-ordered grid.
    
    # Let's use a 14x15 grid. 14*15 = 210. Remove 10 points from the corners to maintain symmetry?
    # Or just use a 15x15 grid and remove the 25 points with the largest distance from the center?
    # That would leave a cluster in the center. Bad.
    
    # Let's use a 14x15 grid.
    m_x = 14
    m_y = 15
    grid_points = []
    for i in range(m_x):
        for j in range(m_y):
            x = (i + 0.5) / m_x - 0.5
            y = (j + 0.5) / m_y - 0.5
            grid_points.append([x, y])
            
    # We have 210 points. We need 200.
    # Remove the 10 points with the largest distance from the center?
    # Or just take the first 200?
    # Let's take the first 200 in row-major order. This will leave the last 10 missing from the bottom right.
    # This is asymmetric.
    
    # Let's use a 15x15 grid and remove the 25 points that are furthest from the center?
    # No, that creates a hole in the middle.
    
    # Let's use a 15x15 grid and remove the 25 points that are closest to the corners?
    # That would make the corners empty.
    
    # Let's just use a 14x14 grid (196 points) and add 4 points in the center of the empty spaces?
    # This is getting complicated.
    
    # Simplest robust initialization:
    # Use a 15x15 grid.
    # Randomly select 200 points.
    # To avoid large gaps, we can use a stratified random sample.
    # Divide the square into 200 cells? No.
    
    # Let's use the "Sunflower" pattern mapped to the square as discussed in the previous thought,
    # but implemented correctly.
    # r_k = sqrt(k/N) * R_max
    # theta_k = 2*pi*k/golden_ratio
    # x = r * cos(theta)
    # y = r * sin(theta)
    # If we set R_max = 0.5 * sqrt(2), the points will be in a circle that touches the corners.
    # Points outside the square (|x|>0.5 or |y|>0.5) need to be handled.
    # We can clip them? Clipping causes overlaps.
    # We can reflect them?
    
    # Let's go with the jittered grid of 15x15 and remove 25 points.
    # To maintain symmetry, remove points in a pattern.
    # Or just use a 14x15 grid and remove 10 points from the last row?
    
    # Let's use a 14x15 grid.
    m_x = 14
    m_y = 15
    points = np.zeros((N, 2))
    idx = 0
    for i in range(m_x):
        for j in range(m_y):
            if idx >= N:
                break
            x = (i + 0.5) / m_x - 0.5
            y = (j + 0.5) / m_y - 0.5
            # Add small jitter to break symmetry
            points[idx, 0] = x + (np.random.rand() - 0.5) * 0.005
            points[idx, 1] = y + (np.random.rand() - 0.5) * 0.005
            idx += 1
        if idx >= N:
            break

    # 2. Optimization: L-BFGS-B with soft potential
    # Define the energy function
    def energy(x_flat):
        if time.time() > deadline:
            return 1e10  # Large penalty if we run out of time
        
        pts = x_flat.reshape(N, 2)
        
        # Calculate pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        
        # Soft repulsive potential: sum of 1/d^2
        # To avoid division by zero, add a small epsilon
        # We only care about pairs that are close, but for smoothness, we sum all.
        # 1/d^2 is very steep. Maybe 1/d^4?
        # Let's use a standard repulsive potential.
        # E_rep = sum_{i<j} 1 / (d_ij^2 + eps)
        
        # This can be large if points are very close.
        # Let's use a softer potential that is still smooth.
        # E_rep = sum_{i<j} exp(-d_ij^2 / sigma^2) ? No, that's attractive.
        # We want repulsive.
        # E_rep = sum_{i<j} 1 / (d_ij + eps) ?
        
        # Let's use the "overlap" penalty approach.
        # We want to maximize the minimum distance.
        # Minimizing the sum of squared overlaps for a target radius r is good.
        # But r is not fixed.
        
        # Alternative: Minimize the sum of 1/d_ij^2.
        # This drives all pairs apart.
        
        eps = 1e-8
        # Mask the diagonal
        mask = ~np.eye(N, dtype=bool)
        dist_sq_masked = dist_sq[mask]
        
        # Sum of 1/d^2
        # This is O(N^2)
        e_rep = np.sum(1.0 / (dist_sq_masked + eps))
        
        # Wall penalty
        # Distance to walls
        d_wall_x_neg = 0.5 - pts[:, 0]
        d_wall_x_pos = 0.5 + pts[:, 0]
        d_wall_y_neg = 0.5 - pts[:, 1]
        d_wall_y_pos = 0.5 + pts[:, 1]
        
        # We want to keep points inside.
        # Penalty if distance is small.
        # E_wall = sum 1 / d_wall^2
        # But if a point is outside, d is negative.
        # We should penalize being outside heavily.
        # Let's use a penalty for |d| < threshold?
        # Or just use the same 1/d^2 potential but with absolute distance?
        # If d is negative, 1/d^2 is positive.
        # This will push them away from the wall.
        
        e_wall_x_neg = np.sum(1.0 / (d_wall_x_neg**2 + eps))
        e_wall_x_pos = np.sum(1.0 / (d_wall_x_pos**2 + eps))
        e_wall_y_neg = np.sum(1.0 / (d_wall_y_neg**2 + eps))
        e_wall_y_pos = np.sum(1.0 / (d_wall_y_pos**2 + eps))
        
        e_wall = e_wall_x_neg + e_wall_x_pos + e_wall_y_neg + e_wall_y_pos
        
        return e_rep + e_wall

    # Initial guess
    x0 = points.flatten()
    
    # Run optimization
    best_points = points.copy()
    best_r = -1.0
    
    def calc_radius(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dist_mat, np.inf)
        min_pair_dist = np.min(dist_mat)
        
        d_b = np.min([
            0.5 - pts[:, 0],
            0.5 + pts[:, 0],
            0.5 - pts[:, 1],
            0.5 + pts[:, 1]
        ], axis=0)
        min_boundary_dist = np.min(d_b)
        
        return min(min_pair_dist / 2.0, min_boundary_dist)
    
    current_r = calc_radius(points)
    if current_r > best_r:
        best_r = current_r
        best_points = points.copy()
    
    try:
        # Use L-BFGS-B
        # Bounds: points must be in [-0.5, 0.5]
        bounds = [(-0.5, 0.5)] * (N * 2)
        
        result = minimize(
            energy,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'maxfun': 10000, 'disp': False}
        )
        
        if result.success:
            new_points = result.x.reshape(N, 2)
            new_r = calc_radius(new_points)
            if new_r > best_r:
                best_r = new_r
                best_points = new_points.copy()
                
    except Exception:
        pass
    
    return best_points
# EVOLVE-BLOCK-END
