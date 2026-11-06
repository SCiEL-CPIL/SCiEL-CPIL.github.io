"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(pts):
    """Compute the largest common radius for given centers."""
    n = len(pts)
    if n < 2:
        return 0.0
    # Boundary distance
    boundary = 0.5 - np.abs(pts).max(axis=1)
    r_boundary = np.min(boundary)
    
    # Pairwise distances (vectorized)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dist, np.inf)
    min_pairwise = np.min(dist)
    r_pairwise = min_pairwise / 2.0
    
    return min(r_boundary, r_pairwise)

def _generate_hex_grid(n_points):
    """Generate a hexagonal grid packing for n_points in [-0.5, 0.5]^2."""
    # For hexagonal packing, we want to fit as many points as possible
    # in a hexagonal arrangement within the unit square.
    
    # Start with a hexagonal lattice
    # The optimal hexagonal packing has spacing d between centers
    # For a square of side 1, we can fit approximately sqrt(4*n*sqrt(3)/3) / 2 rows
    
    # Let's try different numbers of rows and columns
    best_pts = None
    best_r = 0.0
    
    # Try various configurations
    for rows in range(10, 25):
        cols = int(np.ceil(n_points / rows))
        if cols < 1:
            continue
        
        # Hexagonal spacing
        # For hexagonal packing, horizontal spacing = d, vertical spacing = d * sqrt(3)/2
        # Alternate rows are offset by d/2
        
        # We need to fit rows * sqrt(3)/2 * d <= 1 and cols * d <= 1
        # So d <= min(1/cols, 2/(rows*sqrt(3)))
        d = min(1.0 / cols, 2.0 / (rows * np.sqrt(3)))
        
        # Generate points
        pts = []
        for i in range(rows):
            y = -0.5 + d * np.sqrt(3) / 2 * (i + 0.5)
            offset = d / 2 if i % 2 == 1 else 0
            for j in range(cols):
                x = -0.5 + d * (j + 0.5) + offset
                pts.append([x, y])
        
        pts = np.array(pts[:n_points])
        
        if len(pts) < n_points:
            continue
        
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(n_points)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = grid[:n_points]
        best_r = _compute_radius(best_pts)
    
    return best_pts, best_r

def _objective_and_grad(flat_pts, n_points):
    """Objective function: negative of the minimum radius (to minimize)."""
    pts = flat_pts.reshape(-1, 2)
    n = len(pts)
    
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Boundary distances
    boundary = 0.5 - np.abs(pts)
    boundary_min = np.min(boundary)
    
    # Find the minimum pairwise distance
    min_dist_idx = np.unravel_index(np.argmin(dist), dist.shape)
    min_dist = dist[min_dist_idx]
    
    # The radius is min(boundary_min, min_dist/2)
    # We want to maximize this, so we minimize -radius
    
    if min_dist / 2 < boundary_min:
        r = min_dist / 2
        # Gradient with respect to the two points involved
        grad = np.zeros_like(flat_pts)
        i, j = min_dist_idx
        diff_ij = pts[i] - pts[j]
        norm = np.linalg.norm(diff_ij)
        if norm > 1e-12:
            grad[2*i:2*i+2] = -0.5 * diff_ij / norm
            grad[2*j:2*j+2] = 0.5 * diff_ij / norm
    else:
        r = boundary_min
        # Gradient from boundary
        grad = np.zeros_like(flat_pts)
        # Find which point has the minimum boundary distance
        boundary_flat = np.min(boundary, axis=1)
        min_boundary_idx = np.argmin(boundary_flat)
        for k in range(2):
            if pts[min_boundary_idx, k] > 0:
                grad[2*min_boundary_idx + k] = -1.0
            elif pts[min_boundary_idx, k] < 0:
                grad[2*min_boundary_idx + k] = 1.0
            else:
                grad[2*min_boundary_idx + k] = 0.0
    
    return -r, grad

def construct():
    """Construct centers for N equal circles in [-0.5, 0.5]^2."""
    start_time = time.time()
    deadline = start_time + 135.0  # 150s limit, keep 15s margin
    
    # Generate initial hexagonal grid
    pts, r0 = _generate_hex_grid(N)
    best_pts = pts.copy()
    best_r = r0
    
    # Use L-BFGS-B to optimize
    flat_pts = pts.flatten()
    
    try:
        result = minimize(
            _objective_and_grad,
            flat_pts,
            method='L-BFGS-B',
            jac=True,
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000}
        )
        
        if result.success or result.fun < -best_r:
            new_pts = result.x.reshape(-1, 2)
            new_r = _compute_radius(new_pts)
            if new_r > best_r:
                best_pts = new_pts
                best_r = new_r
        
        # Check time
        if time.time() > deadline - 30:
            return best_pts
        
        # Do another refinement pass with tighter tolerance
        result2 = minimize(
            _objective_and_grad,
            best_pts.flatten(),
            method='L-BFGS-B',
            jac=True,
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 10000}
        )
        
        if result2.success or result2.fun < -best_r:
            new_pts = result2.x.reshape(-1, 2)
            new_r = _compute_radius(new_pts)
            if new_r > best_r:
                best_pts = new_pts
                best_r = new_r
        
    except Exception as e:
        # If optimization fails, return the initial grid
        pass
    
    # Ensure points are within bounds
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts
# EVOLVE-BLOCK-END
