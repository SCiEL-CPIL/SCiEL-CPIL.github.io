"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import KDTree

N = 300

# EVOLVE-BLOCK-START
import numpy as np

def _init_hexagonal(N, R=1.0):
    """
    Initialize points in a hexagonal grid pattern.
    """
    points = []
    # Estimate side length of hexagon
    # Area of hexagon with side a is (3*sqrt(3)/2) * a^2
    # N * (3*sqrt(3)/2) * a^2 ~ pi * R^2
    # a ~ sqrt(pi * R^2 / (N * 3*sqrt(3)/2))
    a = np.sqrt(np.pi * R**2 / (N * 1.5 * np.sqrt(3)))
    
    # Generate hex grid
    # Center at (0,0)
    # Hex grid spacing:
    # dx = 1.5 * a
    # dy = sqrt(3) * a
    # Offset for odd columns
    
    max_radius = R
    # Determine grid range
    # x from -max_radius to max_radius
    # y from -max_radius to max_radius
    
    # We'll generate a large grid and filter
    grid_points = []
    # Number of columns
    num_cols = int(2 * max_radius / (1.5 * a)) + 2
    num_rows = int(2 * max_radius / (np.sqrt(3) * a)) + 2
    
    for i in range(-num_cols, num_cols + 1):
        x = i * 1.5 * a
        if abs(x) > max_radius:
            continue
        for j in range(-num_rows, num_rows + 1):
            y = j * np.sqrt(3) * a + (0.5 * np.sqrt(3) * a if i % 2 != 0 else 0)
            if x*x + y*y <= max_radius**2:
                grid_points.append([x, y])
                
    grid_points = np.array(grid_points)
    
    # If we have too many, subsample randomly
    if len(grid_points) > N:
        indices = np.random.choice(len(grid_points), N, replace=False)
        grid_points = grid_points[indices]
    elif len(grid_points) < N:
        # Add random points if not enough
        while len(grid_points) < N:
            x, y = np.random.uniform(-max_radius, max_radius, 2)
            if x*x + y*y <= max_radius**2:
                grid_points = np.vstack([grid_points, [[x, y]]])
            else:
                 # Force fit
                 x, y = np.random.uniform(-max_radius*0.5, max_radius*0.5, 2)
                 grid_points = np.vstack([grid_points, [[x, y]]])
                
    return grid_points[:N]

def _compute_gradients(points, p=10.0):
    """
    Compute gradients of the repulsive potential E = sum 1/d^p + sum 1/(1-r)^p
    """
    pts = points
    n = len(pts)
    grads = np.zeros_like(pts)
    
    # Pairwise distances
    # To save memory, we can do this in chunks or use broadcasting if N is small enough.
    # N=300, 300x300 matrix is fine.
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1.0) # Avoid div by zero, we'll mask later
    dist = np.sqrt(dist_sq)
    
    # Repulsion term: d/dp_i (1/d_ij^p) = -p * diff_ij / d_ij^(p+2)
    # Sum over j
    # inv_dist_pow_p2 = 1 / dist_sq ** ((p+2)/2.0)
    # Gradient contribution from j to i: -p * (p_i - p_j) / d_ij^(p+2)
    # Note: diff[i,j] = p_i - p_j
    
    # Create a weight matrix
    # w_ij = p / dist_ij^(p+2)
    # But careful with diagonal
    mask = np.ones_like(dist_sq, dtype=bool)
    np.fill_diagonal(mask, False)
    
    # Compute weights
    # dist_sq > 0 for all i!=j
    weights = np.zeros_like(dist_sq)
    weights[mask] = p / (dist_sq[mask] ** ((p+2)/2.0))
    
    # grad_i = - sum_j weights[i,j] * (p_i - p_j)
    # diff[i,j] = p_i - p_j
    # So grad_i = - sum_j weights[i,j] * diff[i,j]
    # Wait, d/dp_i (1/|p_i-p_j|^p) = -p * (p_i-p_j) / |p_i-p_j|^(p+2)
    # So yes.
    
    # Sum over j (axis 1)
    # grad = - np.sum(weights[:, :, np.newaxis] * diff, axis=1)
    
    # Boundary term: E_b = sum 1/(1-|p_i|)^p
    # d/dp_i (1/(1-|p_i|)^p) = -p * (1/(1-|p_i|)^(p+1)) * d/dp_i (1-|p_i|)
    # d/dp_i |p_i| = p_i / |p_i|
    # So d/dp_i (1-|p_i|) = -p_i / |p_i|
    # Term = -p * (1/(1-|p_i|)^(p+1)) * (-p_i / |p_i|) = p * p_i / (|p_i| * (1-|p_i|)^(p+1))
    
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    # Avoid div by zero
    norms_safe = np.where(norms < 1e-6, 1e-6, norms)
    
    # Boundary gradient
    # Factor = p / (norms_safe * (1 - norms_safe)^(p+1))
    # But if norms > 1, we are outside. The function is undefined.
    # We assume points are kept inside.
    # If norms close to 1, gradient is huge.
    
    # Only compute for norms < 1
    inside_mask = norms.flatten() < 0.999
    b_grad = np.zeros_like(pts)
    if np.any(inside_mask):
        n_in = norms[inside_mask]
        factor = p / (n_in * (1.0 - n_in.flatten())**(p+1))
        b_grad[inside_mask] = factor[:, np.newaxis] * pts[inside_mask]
        
    # Pairwise gradient
    # grad_pair = - np.sum(weights[:, :, np.newaxis] * diff, axis=1)
    grad_pair = - np.sum(weights[:, :, np.newaxis] * diff, axis=1)
    
    return grad_pair + b_grad

def _optimize_repulsion(points, steps=2000, p=10.0, lr_start=0.01, lr_end=0.0001):
    """
    Gradient descent with momentum on repulsive potential.
    """
    pts = points.copy()
    momentum = np.zeros_like(pts)
    beta = 0.9
    
    # Learning rate schedule
    lrs = np.linspace(lr_start, lr_end, steps)
    
    for t in range(steps):
        grads = _compute_gradients(pts, p=p)
        
        # Update momentum
        momentum = beta * momentum - lrs[t] * grads
        pts = pts + momentum
        
        # Project to unit circle
        norms = np.linalg.norm(pts, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / norms[mask, np.newaxis] * 0.999999
            
    return pts

def _score(points):
    """
    Calculate the score: min(min_pair_dist/2, min_dist_to_boundary)
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dist_sq))
    
    norms = np.linalg.norm(points, axis=1)
    min_boundary_dist = 1.0 - np.max(norms)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)

def construct():
    # Multi-start
    best_points = None
    best_score = -1.0
    
    for trial in range(5):
        # Initialize
        if trial == 0:
            init_pts = _init_hexagonal(N)
        else:
            # Random initialization with slight structure
            init_pts = np.random.uniform(-0.9, 0.9, (N, 2))
            # Project inside
            norms = np.linalg.norm(init_pts, axis=1)
            mask = norms > 0.9
            init_pts[mask] = init_pts[mask] / norms[mask, np.newaxis] * 0.9
            
        # Optimize
        opt_pts = _optimize_repulsion(init_pts, steps=3000, p=8.0, lr_start=0.05, lr_end=0.001)
        
        # Refine with higher p and lower lr
        opt_pts = _optimize_repulsion(opt_pts, steps=2000, p=12.0, lr_start=0.01, lr_end=0.0001)
        
        score = _score(opt_pts)
        if score > best_score:
            best_score = score
            best_points = opt_pts.copy()
            
    # Final polish
    if best_points is not None:
        final_pts = _optimize_repulsion(best_points, steps=1000, p=15.0, lr_start=0.005, lr_end=0.00005)
        final_score = _score(final_pts)
        if final_score > best_score:
            best_points = final_pts
            best_score = final_score

    # Ensure validity
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms >= 1.0
    if np.any(mask):
        best_points[mask] = best_points[mask] / norms[mask, np.newaxis] * 0.999999
        
    return best_points
# EVOLVE-BLOCK-END
