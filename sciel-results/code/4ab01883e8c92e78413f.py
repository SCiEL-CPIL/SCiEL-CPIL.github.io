import numpy as np
import time
from scipy.optimize import minimize

N = 21
TIME_LIMIT = 140  # seconds, leave margin for 150s kill

def _compute_max_radius(centers, w, h):
    """
    Given fixed centers, compute the maximum uniform radius r such that
    all circles of radius r fit in the w x h rectangle and are disjoint.
    This is a structural change: optimize positions only, radii derived.
    """
    n = len(centers)
    # Boundary constraint
    r_boundary = np.min(np.minimum(np.minimum(centers[:, 0], w - centers[:, 0]),
                                   np.minimum(centers[:, 1], h - centers[:, 1])))
    # Pairwise constraint: for uniform radius, d_ij >= 2r => r <= d_ij/2
    r_pair = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((centers[i, 0] - centers[j, 0])**2 + (centers[i, 1] - centers[j, 1])**2)
            r_pair = min(r_pair, d / 2.0)
    return max(0.0, min(r_boundary, r_pair))

def _objective_positions(centers_flat, w, h):
    """
    Objective function for position optimization.
    Minimizes negative sum of radii (which is -N * r_max for uniform radius).
    """
    centers = centers_flat.reshape(N, 2)
    # Check if centers are within bounds (with small tolerance)
    if np.any(centers[:, 0] < -1e-6) or np.any(centers[:, 0] > w + 1e-6) or \
       np.any(centers[:, 1] < -1e-6) or np.any(centers[:, 1] > h + 1e-6):
        return 1e10
    
    r_max = _compute_max_radius(centers, w, h)
    if r_max <= 0:
        return 1e10
    return -N * r_max

def _grid_init(w, h):
    """Generate a grid-like initialization for N circles in w x h rectangle."""
    # Determine grid dimensions based on aspect ratio
    aspect = w / h
    if aspect >= 1:
        cols = int(np.round(np.sqrt(N * aspect)))
        rows = int(np.ceil(N / cols))
    else:
        rows = int(np.round(np.sqrt(N / aspect)))
        cols = int(np.ceil(N / rows))
    
    # Ensure we have at least N cells
    while cols * rows < N:
        if aspect >= 1:
            cols += 1
        else:
            rows += 1
    
    centers = np.zeros((N, 2))
    cell_w = w / cols
    cell_h = h / rows
    
    for i in range(N):
        col = i % cols
        row = i // cols
        centers[i, 0] = (col + 0.5) * cell_w
        centers[i, 1] = (row + 0.5) * cell_h
    
    return centers

def _random_init(w, h):
    """Random initialization of centers."""
    margin = 0.05
    centers = np.random.uniform(margin, np.array([w - margin, h - margin]), (N, 2))
    return centers

def _local_optimize_positions(centers0, w, h, time_budget):
    """
    Local optimization using Nelder-Mead on positions only.
    Radii are derived from positions via _compute_max_radius.
    """
    start = time.time()
    x0 = centers0.flatten()
    
    # Use bounds to keep centers within rectangle
    bounds = [(0, w)] * N + [(0, h)] * N
    
    try:
        res = minimize(_objective_positions, x0, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8,
                                'adaptive': True, 'maxfev': 20000})
    except Exception:
        r_max = _compute_max_radius(centers0, w, h)
        return centers0, r_max, N * r_max
    
    centers_opt = res.x.reshape(N, 2)
    
    # Clamp centers to valid range
    centers_opt[:, 0] = np.clip(centers_opt[:, 0], 0, w)
    centers_opt[:, 1] = np.clip(centers_opt[:, 1], 0, h)
    
    r_max = _compute_max_radius(centers_opt, w, h)
    score = N * r_max
    return centers_opt, r_max, score

def _successive_relaxation_positions(centers, w, h, max_iters=30):
    """
    Successive relaxation: iteratively adjust positions to maximize clearance.
    For each circle, move it towards the center of its largest empty space.
    """
    start = time.time()
    for it in range(max_iters):
        if time.time() - start > 5:
            break
        
        new_centers = centers.copy()
        for i in range(N):
            # Compute gradient-like direction: move away from closest neighbors
            # and towards center of rectangle
            force_x = 0.0
            force_y = 0.0
            
            # Boundary forces
            if centers[i, 0] < 0.1:
                force_x += 1.0
            if centers[i, 0] > w - 0.1:
                force_x -= 1.0
            if centers[i, 1] < 0.1:
                force_y += 1.0
            if centers[i, 1] > h - 0.1:
                force_y -= 1.0
            
            # Neighbor repulsion
            for j in range(N):
                if i == j:
                    continue
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 0.3:  # Only repel when close
                    if d > 1e-9:
                        force_x += dx / d * (0.3 - d)
                        force_y += dy / d * (0.3 - d)
            
            # Update with damping
            new_centers[i, 0] = centers[i, 0] + 0.1 * force_x
            new_centers[i, 1] = centers[i, 1] + 0.1 * force_y
        
        # Clamp
        new_centers[:, 0] = np.clip(new_centers[:, 0], 0, w)
        new_centers[:, 1] = np.clip(new_centers[:, 1], 0, h)
        
        # Check convergence
        if np.max(np.abs(new_centers - centers)) < 1e-8:
            centers = new_centers
            break
        centers = new_centers
    
    return centers

def construct():
    start_time = time.time()
    best_score = -np.inf
    best_config = None
    
    # Try different aspect ratios
    # Perimeter = 2(w+h) = 4 => w+h = 2
    # Try w from 0.5 to 1.5 (h = 2 - w)
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.85, 1.15, 0.75, 1.25, 0.7, 1.3]
    
    for w in aspect_ratios:
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        h = 2.0 - w
        
        # Try multiple initializations
        inits = []
        # Grid init
        centers = _grid_init(w, h)
        inits.append(centers)
        
        # Random inits
        for _ in range(2):
            centers = _random_init(w, h)
            inits.append(centers)
        
        for centers_init in inits:
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            # First do some relaxation to improve initial position
            centers_relaxed = _successive_relaxation_positions(centers_init, w, h, max_iters=20)
            
            # Local optimization
            centers_opt, r_max, score = _local_optimize_positions(centers_relaxed, w, h, 10)
            
            if score > best_score:
                best_score = score
                best_config = np.column_stack((centers_opt, np.full(N, r_max)))
    
    # Final polish: try to improve best config with more iterations
    if best_config is not None and time.time() - start_time < TIME_LIMIT - 20:
        x = best_config[:, 0]
        y = best_config[:, 1]
        r = best_config[:, 2]
        
        # Determine w, h from best config
        # We need to track which aspect ratio gave the best
        # For now, re-optimize with the same w, h
        for w_test in aspect_ratios:
            h_test = 2.0 - w_test
            # Check if best_config fits in this rectangle
            if np.max(x) <= w_test + 1e-6 and np.max(y) <= h_test + 1e-6:
                centers_init = np.column_stack((x, y))
                centers_opt, r_max, score = _local_optimize_positions(centers_init, w_test, h_test, 20)
                if score > best_score:
                    best_score = score
                    best_config = np.column_stack((centers_opt, np.full(N, r_max)))
                break
    
    if best_config is None:
        # Fallback
        w, h = 1.0, 1.0
        centers = _grid_init(w, h)
        r_max = _compute_max_radius(centers, w, h)
        return np.column_stack((centers, np.full(N, r_max)))
    
    return best_config
