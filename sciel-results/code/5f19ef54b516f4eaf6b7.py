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
    
    # 2. Local Optimization: Multi-Bottleneck Repulsion
    # We maximize the minimum distance by applying repulsive forces to all "stressed" pairs
    # (those close to the minimum distance) and handling boundary constraints.
    
    def calc_min_radius(pts):
        # Distance to boundary
        dists_boundary = 1.0 - np.linalg.norm(pts, axis=1)
        min_boundary = np.min(dists_boundary)
        
        # Distance between points
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        return min(min_boundary, min_pair)

    current_r = calc_min_radius(points)
    
    # Optimization parameters
    step_size = 0.05
    max_iter = 20000
    no_improve_count = 0
    threshold = 1e-9
    
    for it in range(max_iter):
        # Calculate all pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        min_dist_val = np.min(dists)
        
        # Identify stressed pairs: those with distance < 2 * min_dist
        # This allows multiple pairs to be resolved simultaneously
        stressed_mask = (dists < 2.0 * min_dist_val) & (dists != np.inf)
        
        # Calculate forces
        # We use unit vectors for repulsion to ensure stable movement
        dists_safe = np.where(stressed_mask, dists, 1.0)
        
        # Unit vector from j to i: (p_i - p_j) / dist
        unit_vecs = diff / dists_safe[:, :, np.newaxis]
        
        # Only keep forces for stressed pairs
        forces_from_pairs = unit_vecs * stressed_mask[:, :, np.newaxis]
        
        # Sum forces on each point
        net_forces = np.sum(forces_from_pairs, axis=1)
        
        # Normalize forces to avoid huge steps
        force_norms = np.linalg.norm(net_forces, axis=1)
        nonzero = force_norms > 1e-10
        net_forces[nonzero] /= force_norms[nonzero, np.newaxis]
        
        # Boundary constraint
        norms = np.linalg.norm(points, axis=1)
        dists_boundary = 1.0 - norms
        min_boundary_val = np.min(dists_boundary)
        
        # Determine dominant constraint
        if min_boundary_val < min_dist_val / 2.0:
            # Boundary is bottleneck
            min_boundary_idx = np.argmin(dists_boundary)
            vec = -points[min_boundary_idx]
            norm_vec = np.linalg.norm(vec)
            if norm_vec > 1e-10:
                points[min_boundary_idx] += (vec / norm_vec) * step_size
            # Apply a small inward force to all points to prevent others from hitting boundary
            points -= points * (step_size * 0.01)
        else:
            # Pairwise is bottleneck
            # Apply the repulsive forces
            points += net_forces * step_size
            
        # Project back into circle
        norms = np.linalg.norm(points, axis=1)
        outside = norms > 1.0
        if np.any(outside):
            points[outside] /= norms[outside, np.newaxis]
            
        new_r = calc_min_radius(points)
        
        if new_r > current_r + threshold:
            current_r = new_r
            no_improve_count = 0
            # Adaptive step size: increase if improving
            if it % 100 == 0:
                step_size = min(step_size * 1.05, 0.1)
        else:
            no_improve_count += 1
            step_size *= 0.95
            if step_size < 1e-6:
                break
            
            # Random perturbation if stuck
            if no_improve_count % 100 == 0:
                points += np.random.normal(0, 0.01, points.shape)
                norms = np.linalg.norm(points, axis=1)
                outside = norms > 1.0
                if np.any(outside):
                    points[outside] /= norms[outside, np.newaxis]
                new_r = calc_min_radius(points)
                if new_r > current_r:
                    current_r = new_r
                    no_improve_count = 0

        if no_improve_count > 1000:
            break

    return points
# EVOLVE-BLOCK-END
