import numpy as np
import time

N = 300

def _init_hex(n):
    """Generate hexagonal lattice points inside unit disk."""
    # Spacing needs to be small enough to fit n points
    # Area of unit disk is pi. Area per circle ~ pi/n. 
    # But we are packing circles, so density is ~ 0.9.
    # Let's just generate a fine grid and pick n closest to center.
    spacing = 0.05
    k = int(1.5 / spacing)
    pts = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                pts.append([x, y])
    pts = np.array(pts)
    # Sort by distance to origin
    r = np.hypot(pts[:, 0], pts[:, 1])
    idx = np.argsort(r, kind="stable")[:n]
    return pts[idx]

def _repulsion_step(pts, r, lr):
    """
    Perform one step of repulsion.
    pts: (N, 2) array of centers.
    r: current target radius.
    lr: learning rate.
    Returns updated pts.
    """
    n = len(pts)
    # Compute pairwise distances
    dx = pts[:, 0:1] - pts[:, 0:1].T
    dy = pts[:, 1:2] - pts[:, 1:2].T
    dist = np.sqrt(dx * dx + dy * dy)
    
    # Avoid division by zero
    dist_safe = np.where(dist < 1e-8, 1e-8, dist)
    
    # Pairwise forces
    # Force is proportional to overlap amount: (2r - dist)
    # Direction is (c_i - c_j) / dist
    overlap = 2.0 * r - dist
    # Only push if overlapping
    overlap = np.where(overlap > 0, overlap, 0.0)
    
    # Normalize direction
    # dir_x = dx / dist, dir_y = dy / dist
    dir_x = dx / dist_safe
    dir_y = dy / dist_safe
    
    # Force on i from j: overlap * dir
    # Sum forces from all j on i
    force_x = np.sum(overlap * dir_x, axis=1)
    force_y = np.sum(overlap * dir_y, axis=1)
    
    # Boundary forces
    # If |c_i| > 1 - r, push in
    r_pts = np.hypot(pts[:, 0], pts[:, 1])
    bound_overlap = r_pts - (1.0 - r)
    bound_overlap = np.where(bound_overlap > 0, bound_overlap, 0.0)
    
    # Direction is -c_i / |c_i|
    # If r_pts is 0, direction is 0
    dir_bx = np.where(r_pts > 1e-8, -pts[:, 0] / np.where(r_pts > 1e-8, r_pts, 1.0), 0.0)
    dir_by = np.where(r_pts > 1e-8, -pts[:, 1] / np.where(r_pts > 1e-8, r_pts, 1.0), 0.0)
    
    force_x += bound_overlap * dir_bx
    force_y += bound_overlap * dir_by
    
    # Update
    pts_new = pts.copy()
    pts_new[:, 0] += lr * force_x
    pts_new[:, 1] += lr * force_y
    
    # Project back inside unit circle (safety)
    r_new = np.hypot(pts_new[:, 0], pts_new[:, 1])
    mask = r_new > 1.0
    if np.any(mask):
        pts_new[mask, 0] *= (1.0 - 1e-6) / r_new[mask]
        pts_new[mask, 1] *= (1.0 - 1e-6) / r_new[mask]
        
    return pts_new

def _compute_score(pts):
    """Compute the actual packing radius."""
    n = len(pts)
    if n < 2:
        return 0.0
    dx = pts[:, 0:1] - pts[:, 0:1].T
    dy = pts[:, 1:2] - pts[:, 1:2].T
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    r = np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(1.0 - r)
    
    return min(min_dist / 2.0, min_boundary)

def construct():
    start_time = time.time()
    time_limit = 115.0 # 5s safety margin
    
    # Initialize
    pts = _init_hex(N)
    current_r = _compute_score(pts)
    
    # Start r slightly below current to allow immediate repulsion if needed, 
    # or start at current and increase.
    # Let's start at 0.5 * current_r and ramp up.
    r = max(0.01, current_r * 0.8)
    
    best_pts = pts.copy()
    best_score = current_r
    
    # Parameters
    lr_init = 0.05
    lr_end = 0.0001
    steps = 10000
    r_increase = 0.00005 # How much to increase r per step
    
    for i in range(steps):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
            
        # Anneal learning rate
        progress = i / steps
        lr = lr_init * (lr_end / lr_init) ** progress
        
        # Try to increase r
        # We want to find the max r such that packing is valid.
        # Strategy: increase r, if it fails (large forces), reduce r or just let forces handle it.
        # Better: Keep r constant for a while, then increase.
        
        pts = _repulsion_step(pts, r, lr)
        
        # Check score periodically
        if i % 100 == 0:
            score = _compute_score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
                
        # Increase r
        # Only increase if the current r is "satisfied" (i.e., forces are small)
        # Or just increase slowly always.
        r += r_increase
        
        # Cap r to not be too optimistic
        if r > 0.06: # Known best is ~0.053
            r = 0.053 + 0.001 * np.sin(i * 0.01) # Small oscillation to escape local optima?
            # Actually, just stop increasing if we hit a plausible max
            if r > 0.055:
                r = 0.055
                
    # Final polish: use the best score found to set r, and run a few more steps with smaller lr
    if time.time() - start_time < time_limit - 10:
        r = best_score
        for i in range(1000):
            if time.time() - start_time > time_limit:
                break
            lr = 0.001 * (1.0 - i/1000.0)
            pts = _repulsion_step(pts, r, lr)
            if i % 50 == 0:
                score = _compute_score(pts)
                if score > best_score:
                    best_score = score
                    best_pts = pts.copy()
                    
    return best_pts
