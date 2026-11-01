import numpy as np

N = 300

# EVOLVE-BLOCK-START
def construct():
    # 1. Initialization: Sunflower (Vogel) Spiral
    # This provides a uniform distribution in the disk.
    i = np.arange(N)
    # Golden angle
    angle = 2.0 * np.pi * i / (1.0 + np.sqrt(5.0))
    # Radius distribution: sqrt(i) ensures uniform area density
    # We scale to fill the unit circle, leaving a small margin for optimization
    r = np.sqrt((i + 0.5) / N)
    
    points = np.column_stack((r * np.cos(angle), r * np.sin(angle)))
    
    # 2. Local Optimization
    # We want to maximize the minimum distance:
    # r_min = min( min_{i<j} ||p_i - p_j|| / 2, min_i (1 - ||p_i||) )
    
    # Precompute indices for efficiency? No, N=300 is small enough for O(N^2) per step.
    # We will use a simple hill-climbing approach with adaptive step size.
    
    current_r = 0.0
    # Initial evaluation
    def calc_min_radius(pts):
        # Distance to boundary
        dists_boundary = 1.0 - np.linalg.norm(pts, axis=1)
        min_boundary = np.min(dists_boundary)
        
        # Distance between points
        # Calculate all pairwise distances
        # Using broadcasting: (N, 1, 2) - (1, N, 2) -> (N, N, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Set diagonal to infinity
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        return min(min_boundary, min_pair)

    current_r = calc_min_radius(points)
    
    # Optimization parameters
    step_size = 0.01
    max_iter = 5000
    no_improve_count = 0
    threshold = 1e-8
    
    for _ in range(max_iter):
        # Identify the bottleneck
        # 1. Find closest pair
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        min_dist_val = np.min(dists)
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx
        
        # 2. Find point closest to boundary
        dists_boundary = 1.0 - np.linalg.norm(points, axis=1)
        min_boundary_val = np.min(dists_boundary)
        min_boundary_idx = np.argmin(dists_boundary)
        
        # Determine which constraint is active
        # If min_dist_val/2 < min_boundary_val, the pair is the bottleneck
        # Else, the boundary is the bottleneck
        
        if min_dist_val / 2.0 < min_boundary_val:
            # Move i and j apart
            vec = points[i] - points[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                direction = vec / norm
                # Move i in direction, j in opposite
                points[i] += direction * step_size * 0.5
                points[j] -= direction * step_size * 0.5
            else:
                # If coincident, random nudge (should not happen with sunflower init)
                points[i] += np.array([step_size, 0])
                points[j] -= np.array([step_size, 0])
        else:
            # Move the point closest to boundary inward
            idx = min_boundary_idx
            vec = -points[idx] # Direction towards center
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                direction = vec / norm
                points[idx] += direction * step_size
            else:
                # At center, move slightly out? No, if at center, boundary is not the min usually unless N=1
                # If at center, any direction is fine, but we want to increase distance to boundary?
                # Actually if at center, distance to boundary is 1.0, which is max.
                # So this case implies the point is NOT at center but close to edge.
                pass

        # Project points back into circle if they went out
        # This is a soft constraint in the move, but we must ensure validity
        norms = np.linalg.norm(points, axis=1)
        outside = norms > 1.0
        if np.any(outside):
            points[outside] /= norms[outside, np.newaxis]
            # If projected, the radius calculation might change, but we stay valid
            
        new_r = calc_min_radius(points)
        
        if new_r > current_r + threshold:
            current_r = new_r
            no_improve_count = 0
            # Maybe increase step size slightly if we are improving fast?
            # step_size = min(step_size * 1.05, 0.05)
        else:
            no_improve_count += 1
            # Reduce step size
            step_size *= 0.9
            if step_size < 1e-6:
                break
            
        if no_improve_count > 50:
            break
            
    return points
# EVOLVE-BLOCK-END
