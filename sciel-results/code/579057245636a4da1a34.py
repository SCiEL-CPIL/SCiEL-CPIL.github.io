import numpy as np
import time

N = 300

def construct():
    # 1. Initialization: Hexagonal Lattice
    # We generate a hexagonal grid and select points.
    # To get a good start, we estimate the optimal spacing.
    # r ~ 0.053 => d ~ 0.106
    # spacing = d
    spacing = 0.106
    max_k = int(1.0 / spacing) + 3
    points = []
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3) / 2
            if x*x + y*y <= 1.0:
                points.append([x, y])
    
    points = np.array(points)
    
    # If we have fewer than N points, increase density (decrease spacing)
    # If we have more, we select N.
    # Selection strategy: Points closest to the center are usually part of the optimal packing.
    dists_to_center = np.hypot(points[:, 0], points[:, 1])
    
    # If we have exactly N, use them.
    # If more, pick N closest to center.
    # If fewer, we need to regenerate with smaller spacing.
    
    while len(points) < N:
        spacing *= 0.95
        points = []
        max_k = int(1.0 / spacing) + 3
        for j in range(-max_k, max_k + 1):
            for i in range(-max_k, max_k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3) / 2
                if x*x + y*y <= 1.0:
                    points.append([x, y])
        points = np.array(points)
        
    if len(points) > N:
        idx = np.argsort(dists_to_center)[:N]
        pts = points[idx].copy()
    else:
        pts = points.copy()

    # 2. Optimization
    # Force-directed relaxation
    # We want to maximize r = min(min_ij d_ij/2, min_i (1 - |c_i|))
    
    def compute_radius(p):
        # p is (N, 2)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        boundary_dists = 1.0 - np.linalg.norm(p, axis=1)
        min_bound = np.min(boundary_dists)
        return min(min_dist / 2.0, min_bound)

    best_r = compute_radius(pts)
    best_pts = pts.copy()
    
    start_time = time.time()
    deadline = 100.0 # Leave 20s for safety
    
    # Phase 1: Coarse
    lr = 0.01
    iters = 20000
    for it in range(iters):
        if time.time() - start_time > deadline:
            break
        
        # Calculate distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        
        min_dist = np.min(dists)
        boundary_dists = 1.0 - np.linalg.norm(pts, axis=1)
        min_bound = np.min(boundary_dists)
        
        r = min(min_dist / 2.0, min_bound)
        
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
            
        # Determine active constraints
        # Margin to include slightly non-active constraints to smooth the landscape
        margin = 0.1 * r + 1e-4
        threshold_dist = 2 * r + margin
        threshold_bound = r + margin
        
        mask_pairs = dists < threshold_dist
        mask_bound = boundary_dists < threshold_bound
        
        forces = np.zeros_like(pts)
        
        # Pairwise forces: push apart
        # Use constant force for simplicity and stability
        # Direction is diff / dists
        # Avoid division by zero
        inv_dists = np.where(mask_pairs, 1.0 / (dists + 1e-9), 0.0)
        # Force on i from j is (c_i - c_j) * inv_dists_ij
        # Sum over j
        forces += np.sum(diff * inv_dists[:, :, np.newaxis], axis=1)
        
        # Boundary forces: push towards center
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit_vecs = pts / norms
        # Force direction is -unit_vecs (towards origin)
        # Magnitude 1 for active
        b_forces = np.where(mask_bound[:, np.newaxis], -unit_vecs, 0.0)
        forces += b_forces
        
        # Update
        pts += lr * forces
        
        # Optional: Random perturbation to escape local minima
        if it % 500 == 0 and it > 0:
            pts += np.random.normal(0, 0.005, pts.shape)

    # Phase 2: Fine
    lr = 0.001
    iters = 20000
    for it in range(iters):
        if time.time() - start_time > deadline:
            break
            
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        
        min_dist = np.min(dists)
        boundary_dists = 1.0 - np.linalg.norm(pts, axis=1)
        min_bound = np.min(boundary_dists)
        
        r = min(min_dist / 2.0, min_bound)
        
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
            
        margin = 0.05 * r + 1e-5
        threshold_dist = 2 * r + margin
        threshold_bound = r + margin
        
        mask_pairs = dists < threshold_dist
        mask_bound = boundary_dists < threshold_bound
        
        forces = np.zeros_like(pts)
        
        inv_dists = np.where(mask_pairs, 1.0 / (dists + 1e-9), 0.0)
        forces += np.sum(diff * inv_dists[:, :, np.newaxis], axis=1)
        
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit_vecs = pts / norms
        b_forces = np.where(mask_bound[:, np.newaxis], -unit_vecs, 0.0)
        forces += b_forces
        
        pts += lr * forces

    # Phase 3: Ultra-fine
    lr = 0.0001
    iters = 10000
    for it in range(iters):
        if time.time() - start_time > deadline:
            break
            
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        
        min_dist = np.min(dists)
        boundary_dists = 1.0 - np.linalg.norm(pts, axis=1)
        min_bound = np.min(boundary_dists)
        
        r = min(min_dist / 2.0, min_bound)
        
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
            
        margin = 0.01 * r + 1e-6
        threshold_dist = 2 * r + margin
        threshold_bound = r + margin
        
        mask_pairs = dists < threshold_dist
        mask_bound = boundary_dists < threshold_bound
        
        forces = np.zeros_like(pts)
        
        inv_dists = np.where(mask_pairs, 1.0 / (dists + 1e-9), 0.0)
        forces += np.sum(diff * inv_dists[:, :, np.newaxis], axis=1)
        
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit_vecs = pts / norms
        b_forces = np.where(mask_bound[:, np.newaxis], -unit_vecs, 0.0)
        forces += b_forces
        
        pts += lr * forces

    return best_pts
