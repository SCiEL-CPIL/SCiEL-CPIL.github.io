"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_score(pts):
    """Compute the packing radius: min of half-min-distance and min(1-|c_i|)."""
    n = len(pts)
    if n < 2:
        return 0.0
    # Pairwise distances
    dx = pts[:, 0:1] - pts[:, 0:1].T
    dy = pts[:, 1:2] - pts[:, 1:2].T
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    # Distance to boundary
    r = np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(1.0 - r)
    return min(min_dist / 2.0, min_boundary)

def _generate_hex_init(n):
    """Generate hexagonal lattice points clipped to unit disk, select n closest to center."""
    # Best known r ~ 0.053, spacing ~ 0.106. 
    # We want a grid that allows for ~300 points.
    # Area of unit circle is pi ~ 3.14. Area per circle is pi*r^2.
    # Total area ~ 300 * pi * r^2. 
    # r ~ 0.053 -> area ~ 300 * 3.14 * 0.0028 ~ 2.6. 
    # So packing density is high.
    # Spacing s = 2r ~ 0.106.
    # Let's generate a grid with spacing slightly larger than optimal to ensure we can pick n points, 
    # or generate a dense grid and pick the best n.
    
    # A good strategy: generate a hexagonal grid covering the disk, 
    # then select the n points that maximize the minimum distance locally? 
    # Simpler: just generate a grid with spacing ~ 0.1, filter to disk, 
    # and if we have more than n, keep the ones closest to center.
    
    spacing = 0.10
    k = int(1.1 / spacing) + 2
    points = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            points.append([x, y])
    points = np.array(points)
    r = np.hypot(points[:, 0], points[:, 1])
    mask = r <= 1.0 - spacing / 2.0
    points = points[mask]
    
    if len(points) < n:
        # If we don't have enough, we need a finer grid or different init
        # But for N=300, spacing 0.1 gives approx area 3.14 / (0.1^2 * 0.866) ~ 360 points. So it should be enough.
        pass

    # Sort by distance to center and take the first n
    r = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(r, kind="stable")[:n]
    return points[idx]

def _optimize_sparse(pts, time_limit):
    """
    Optimize packing using sparse gradient ascent on active constraints.
    """
    start_time = time.time()
    n = len(pts)
    pts = pts.copy().astype(np.float64)
    
    best_score = _compute_score(pts)
    best_pts = pts.copy()
    
    # Parameters
    step_size = 0.01
    step_decay = 0.999
    active_margin = 0.01 # Points within this distance of the min gap are "active"
    max_iters = 100000
    
    for it in range(max_iters):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 5.0:
            break
        
        # Compute distances
        dx = pts[:, 0:1] - pts[:, 0:1].T
        dy = pts[:, 1:2] - pts[:, 1:2].T
        dist = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist, np.inf)
        
        r = np.hypot(pts[:, 0], pts[:, 1])
        boundary_gaps = 1.0 - r
        
        min_dist = np.min(dist)
        min_boundary = np.min(boundary_gaps)
        current_r = min(min_dist / 2.0, min_boundary)
        
        if current_r > best_score:
            best_score = current_r
            best_pts = pts.copy()
            
        # Identify active pairs and points
        # Active pairs: dist < min_dist + active_margin
        active_pairs_mask = dist < (min_dist + active_margin)
        np.fill_diagonal(active_pairs_mask, False)
        
        # Active boundary points: boundary_gap < min_boundary + active_margin
        active_bound_mask = boundary_gaps < (min_boundary + active_margin)
        
        # If no active points, we are stuck or converged
        if not np.any(active_pairs_mask) and not np.any(active_bound_mask):
            break
            
        grad = np.zeros_like(pts)
        
        # Process active pairs
        iu, ju = np.where(active_pairs_mask)
        for idx in range(len(iu)):
            i, j = iu[idx], ju[idx]
            d = dist[i, j]
            if d < 1e-10:
                continue
            # Force is repulsive
            # grad_i += (c_i - c_j) / d
            # grad_j += (c_j - c_i) / d
            vec = (pts[i] - pts[j]) / d
            grad[i] += vec
            grad[j] -= vec
            
        # Process active boundary points
        active_indices = np.where(active_bound_mask)[0]
        for i in active_indices:
            ri = r[i]
            if ri < 1e-10:
                # If at center, push randomly? Or just skip. 
                # Center is usually not a boundary constraint unless r is huge.
                continue
            # Force is outward from center? No, boundary constraint is 1-|c| >= r.
            # We want to increase 1-|c|, so we want to move c towards center.
            # Gradient of (1 - |c|) wrt c is -c/|c|.
            # So we add -c/|c| to the gradient.
            vec = -pts[i] / ri
            grad[i] += vec
            
        # Normalize gradient? 
        # The magnitude of the gradient depends on the number of active neighbors.
        # It's better to normalize the total step to a fixed length to avoid huge jumps.
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad /= grad_norm
            
        # Update
        pts_new = pts + step_size * grad
        
        # Ensure points stay inside the unit circle
        r_new = np.hypot(pts_new[:, 0], pts_new[:, 1])
        if np.any(r_new >= 1.0):
            # Project back
            mask_out = r_new >= 1.0
            pts_new[mask_out] *= (0.999 / r_new[mask_out])
            
        # Check if the new configuration is valid (no overlaps, inside circle)
        # The score computation will handle this, but we can do a quick check.
        score_new = _compute_score(pts_new)
        
        # Accept if better, or with some probability if worse (annealing)
        # For simplicity, accept if better. If not, we might need to perturb.
        if score_new > best_score:
            pts = pts_new
        else:
            # If we didn't improve, try a small random perturbation on active points to escape local minima
            # This is a simple form of simulated annealing / random restart on local scale
            if it % 10 == 0:
                # Perturb active points
                active_mask = np.zeros(n, dtype=bool)
                active_mask[iu] = True
                active_mask[ju] = True
                active_mask[active_indices] = True
                
                noise = np.random.normal(0, 0.005, size=pts.shape)
                noise[~active_mask] = 0
                pts_perturbed = pts + noise
                r_p = np.hypot(pts_perturbed[:, 0], pts_perturbed[:, 1])
                if np.all(r_p < 1.0):
                    score_p = _compute_score(pts_perturbed)
                    if score_p > best_score:
                        pts = pts_perturbed
                    else:
                        # If perturbation didn't help, keep original pts but decay step size
                        pass
            
        step_size *= step_decay
        if step_size < 1e-6:
            # Reset step size if it gets too small, maybe with a larger perturbation
            step_size = 0.01
            # Random restart?
            # For now, just break or continue with small steps
            pass

    return best_pts, best_score

def construct():
    """
    Construct N=300 equal circles in unit disk.
    Uses hexagonal lattice init + sparse gradient optimization.
    """
    start_time = time.time()
    total_time_limit = 115.0  # Safety margin before 120s kill
    
    # Generate initial hexagonal lattice
    pts = _generate_hex_init(N)
    
    # Main optimization phase
    main_pts, main_score = _optimize_sparse(pts, total_time_limit)
    
    # Final validation: ensure all points are strictly inside
    r = np.hypot(main_pts[:, 0], main_pts[:, 1])
    for i in range(len(main_pts)):
        if r[i] >= 1.0:
            main_pts[i] *= 0.999 / r[i]
    
    return main_pts
# EVOLVE-BLOCK-END
