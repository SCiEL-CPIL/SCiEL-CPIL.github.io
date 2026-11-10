import numpy as np
import time

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Initialization
    m = 14
    ticks = (np.arange(m) + 0.5) / m - 0.5
    xx, yy = np.meshgrid(ticks, ticks)
    pts = np.column_stack((xx.ravel(), yy.ravel()))
    
    # Add 4 random points
    extra = np.random.uniform(-0.5, 0.5, size=(4, 2))
    pts = np.vstack([pts, extra])
    
    # Jitter
    pts += np.random.uniform(-0.005, 0.005, size=pts.shape)
    pts = np.clip(pts, -0.5, 0.5)
    
    best_radius = 0.0
    best_pts = pts.copy()
    
    alpha = 0.01
    decay = 0.995
    
    while time.time() < deadline:
        # Calculate pairwise differences and distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero
        np.fill_diagonal(dist, np.inf)
        
        # Calculate repulsive forces: F_i = sum_j (c_i - c_j) / dist_ij^3
        # diff is c_i - c_j.
        # Force term: diff / dist^3
        # We need to sum over j for each i.
        
        # dist^3
        dist_cubed = dist ** 3
        dist_cubed[dist_cubed == 0] = 1.0 # Safety
        
        # force_matrix[i, j] = diff[i, j] / dist_cubed[i, j]
        # Note: diff[i, j] is vector from j to i.
        # So force[i, j] is repulsive force on i from j.
        force_matrix = diff / dist_cubed[:, :, np.newaxis]
        
        # Sum over j (axis=1) to get total force on each i
        forces = np.sum(force_matrix, axis=1)
        
        # Boundary repulsion
        # Add forces pushing away from boundaries
        # d_left = x + 0.5. Force should be +x if close to left.
        # Let's add a constant push from boundaries, or use a potential.
        # Simple: if x < -0.5 + margin, push right.
        # We can just clamp and let the repulsion handle it, but boundary constraints are hard.
        # Let's add a linear repulsion from boundaries.
        # F_boundary_x = k * (0.5 - |x|) ? No.
        # If x is close to -0.5, we want force +x.
        # Force_x = k * (x + 0.5) if x < 0 ?
        # Let's use: F_x = k * (0.5 - |x|) is not a force, it's a distance.
        # We want to push towards center.
        # F_x = k * x ? No, that pushes to 0.
        # We want to push away from boundaries.
        # If x > 0, push left? No, if x is close to 0.5, push left.
        # If x < 0, push right? If x is close to -0.5, push right.
        # So F_x = -k * x ? No.
        # Let's just clamp. The repulsion from other circles will push them in.
        # But if a circle is alone in a corner, it will stay there.
        # We should add a small bias towards the center?
        # Or just rely on the fact that we want to maximize min distance, which usually fills the space.
        
        # Apply forces
        pts += alpha * forces
        
        # Clamp
        pts = np.clip(pts, -0.5, 0.5)
        
        # Calculate radius
        min_dist_neighbor = np.min(dist, axis=1)
        d_left = pts[:, 0] + 0.5
        d_right = 0.5 - pts[:, 0]
        d_bottom = pts[:, 1] + 0.5
        d_top = 0.5 - pts[:, 1]
        d_boundary = np.min(np.stack([d_left, d_right, d_bottom, d_top]), axis=1)
        
        current_radius = np.min(np.minimum(min_dist_neighbor, d_boundary)) / 2.0
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_pts = pts.copy()
            
        alpha *= decay
        
    return best_pts
