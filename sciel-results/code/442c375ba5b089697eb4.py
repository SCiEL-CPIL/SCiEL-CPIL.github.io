"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np

def _generate_hex_grid(n):
    """
    Generate a hexagonal lattice and select n points that are well-distributed
    and close to the center to provide a good warm start.
    """
    # Estimate spacing. Area of square is 1.
    # Hexagonal packing density is pi/sqrt(12) ~ 0.79.
    # Area per circle ~ 1/n.
    # r ~ sqrt(1/(n * density * 4)) ... roughly 0.035 for n=200.
    # Spacing d = 2r ~ 0.07.
    
    # We will generate a grid with spacing slightly smaller than optimal
    # and then select the best n points.
    
    # Try a range of spacings
    best_points = None
    best_score = -1
    
    for d in np.linspace(0.06, 0.09, 20):
        # Hex grid generation
        # Rows are offset
        # Vertical distance between rows: d * sqrt(3) / 2
        # Horizontal distance: d
        # Offset for odd rows: d / 2
        
        # Estimate grid size
        # y range [-0.5, 0.5]
        # num_rows ~ 1.0 / (d * sqrt(3) / 2)
        num_rows = int(1.0 / (d * np.sqrt(3) / 2)) + 2
        num_cols = int(1.0 / d) + 2
        
        points = []
        
        # Center the grid
        # Start y at -0.5 + offset
        # Let's generate all points in a larger box and filter
        
        for j in range(-num_rows, num_rows + 1):
            y = j * d * np.sqrt(3) / 2
            if y < -0.5 or y > 0.5:
                continue
            offset = 0.0
            if j % 2 != 0:
                offset = d / 2.0
            
            for i in range(-num_cols, num_cols + 1):
                x = i * d + offset
                if x < -0.5 or x > 0.5:
                    continue
                points.append([x, y])
        
        if len(points) < n:
            continue
            
        points = np.array(points)
        
        # We have more than n points. We want to select n points.
        # Strategy: To maximize the minimum distance, we should avoid boundary points
        # if possible, or select points that form a compact cluster.
        # A simple heuristic: Sort by distance from center (0,0) and take the closest n?
        # This might create a hole in the middle if the grid is sparse? No, hex grid is dense.
        # Taking closest to center ensures we stay away from boundaries, which is good
        # because boundary points limit the radius more severely if not centered perfectly.
        
        dists_to_center = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists_to_center)
        selected = points[idx[:n]]
        
        # Calculate current score for this selection
        # Score = min(min_pair_dist / 2, min_boundary_dist)
        
        # Pairwise distances
        diff = selected[:, np.newaxis, :] - selected[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        # Boundary distances
        boundary_dists = np.minimum(0.5 - np.abs(selected[:, 0]), 0.5 - np.abs(selected[:, 1]))
        min_boundary = np.min(boundary_dists)
        
        score = min(min_pair / 2, min_boundary)
        
        if score > best_score:
            best_score = score
            best_points = selected.copy()
            
    if best_points is None:
        # Fallback: simple grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n]
        
    return best_points

def construct():
    # 1. Warm start
    points = _generate_hex_grid(N)
    
    # 2. Optimize using Projected Gradient Ascent on "Feasibility Radius"
    # We want to maximize r such that:
    # d_ij >= 2r for all i<j
    # b_i >= r for all i
    
    # We will iteratively increase r and adjust points to satisfy constraints.
    
    # Initial radius estimate
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    boundary_dists = np.minimum(0.5 - np.abs(points[:, 0]), 0.5 - np.abs(points[:, 1]))
    min_boundary = np.min(boundary_dists)
    
    r = min(min_pair / 2, min_boundary)
    
    # Optimization loop
    # We use a "soft" penalty approach.
    # Force F_i = sum_{j != i} (1 / d_ij) * (points_i - points_j) * max(0, 2r - d_ij)
    # This pushes apart points that are too close.
    
    max_iter = 2000
    lr = 0.001  # Learning rate
    r_increment = 0.0001  # How much to try to increase r
    
    for _ in range(max_iter):
        # Compute pairwise distances and forces
        # To save memory and time, we can do this in chunks or use vectorization carefully.
        # N=200, so 200x200 matrix is 40k elements, which is fine.
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=-1)
        dists = np.sqrt(dists_sq)
        np.fill_diagonal(dists, np.inf)
        np.fill_diagonal(dists_sq, np.inf)
        
        # Constraint violation for pairs: v_ij = max(0, 2*r - d_ij)
        # We want to minimize sum v_ij^2 or similar.
        # Gradient of v_ij w.r.t points_i:
        # If d_ij < 2r, grad = (2r - d_ij) * (points_i - points_j) / d_ij
        
        # Compute forces
        # Factor = max(0, 2*r - d_ij) / d_ij
        # Note: dists has inf on diagonal, so factor will be 0 there.
        # But 2*r - inf is -inf, max(0, -inf) is 0.
        # We need to be careful with division by zero on diagonal, but inf handles it.
        
        # Mask for violations
        mask = dists < 2 * r
        # Factor for force calculation
        # factor_ij = (2*r - d_ij) / d_ij if d_ij < 2r else 0
        factor = np.zeros_like(dists)
        factor[mask] = (2 * r - dists[mask]) / dists[mask]
        
        # Force on i is sum_j factor_ij * (points_i - points_j)
        # diff = points_i - points_j
        # forces = sum_j factor_ij * diff_ij
        # This can be computed as:
        # forces = factor @ diff ? No.
        # factor is N x N. diff is N x N x 2.
        # forces[i] = sum_j factor[i,j] * (points[i] - points[j])
        # forces = factor @ points - (factor @ ones) * points ?
        # Let's expand:
        # F_i = sum_j f_ij P_i - sum_j f_ij P_j
        # F_i = (sum_j f_ij) P_i - sum_j f_ij P_j
        
        row_sums = np.sum(factor, axis=1)
        # sum_j f_ij P_j is (factor @ points)
        weighted_points = factor @ points
        
        pair_forces = row_sums[:, np.newaxis] * points - weighted_points
        
        # Boundary forces
        # If point is too close to boundary (b_i < r), push inward.
        # b_i = min(0.5 - |x_i|, 0.5 - |y_i|)
        # We can approximate gradient of b_i.
        # If x_i > 0, b_i depends on 0.5 - x_i. Grad wrt x is -1.
        # We want to increase b_i, so we move in direction of grad.
        # If b_i < r, we add a force.
        
        boundary_forces = np.zeros_like(points)
        
        # Calculate boundary distances
        bx = 0.5 - np.abs(points[:, 0])
        by = 0.5 - np.abs(points[:, 1])
        
        # Determine which boundary is active
        # If bx < by, x is the limiting factor.
        # If points[i,0] > 0, push left (negative x). If < 0, push right (positive x).
        # Force magnitude proportional to violation (r - b_i).
        
        viol_x = np.maximum(0, r - bx)
        viol_y = np.maximum(0, r - by)
        
        # We only apply force if that dimension is the minimum?
        # Actually, if both are violated, push in both.
        # But if one is smaller, it's the bottleneck.
        # Let's just push in any dimension that is violated.
        
        # Gradient of bx:
        # if x > 0, d/dx (0.5 - x) = -1
        # if x < 0, d/dx (0.5 + x) = 1
        # So grad_bx = -sign(x)
        
        grad_bx = -np.sign(points[:, 0])
        grad_by = -np.sign(points[:, 1])
        
        # If x is 0, sign is 0, which is fine (no preference).
        
        # Force = viol * grad
        boundary_forces[:, 0] += viol_x * grad_bx
        boundary_forces[:, 1] += viol_y * grad_by
        
        total_forces = pair_forces + boundary_forces
        
        # Update points
        points += lr * total_forces
        
        # Project to square
        points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
        points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
        
        # Check if we can increase r
        # Recalculate min distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=-1)
        dists = np.sqrt(dists_sq)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        boundary_dists = np.minimum(0.5 - np.abs(points[:, 0]), 0.5 - np.abs(points[:, 1]))
        min_boundary = np.min(boundary_dists)
        
        current_r = min(min_pair / 2, min_boundary)
        
        # If current_r is significantly larger than r, increase r
        if current_r > r + 1e-6:
            r = current_r
        # If we are stuck (current_r < r), we might need to reduce r or just let forces handle it.
        # The forces should push points apart to satisfy 2r constraint.
        # If they can't, r is too high. But we only increase r when feasible.
        # So r is always feasible at the start of the iteration.
        
        # Early stopping if r doesn't change and forces are small?
        # For simplicity, run all iterations.
        
    # Final calculation of score
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    boundary_dists = np.minimum(0.5 - np.abs(points[:, 0]), 0.5 - np.abs(points[:, 1]))
    min_boundary = np.min(boundary_dists)
    
    return points
# EVOLVE-BLOCK-END
