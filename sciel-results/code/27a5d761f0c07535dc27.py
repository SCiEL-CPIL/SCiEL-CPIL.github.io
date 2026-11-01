"""
Packing 50 equal circles in a unit circle.
Uses hexagonal lattice initialization and vectorized force-based optimization.
"""
import time
import numpy as np

N = 50
# Deadline guard: 150s limit, keep 10s safety margin
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0


def _hex_lattice_init(spacing):
    """
    Generate points on a hexagonal lattice inside the unit disk.
    Returns an array of shape (M, 2) where M >= N if spacing is small enough.
    """
    # Generate a grid large enough to cover the unit circle
    k = int(1.0 / spacing) + 2
    rows = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            rows.append((x, y))
    
    points = np.array(rows)
    radii = np.hypot(points[:, 0], points[:, 1])
    # Keep only points strictly inside (with some margin to allow for movement)
    # We want points that can be moved slightly, so keep those well inside
    mask = radii <= 1.0 - spacing / 2.0
    points = points[mask]
    
    if len(points) < N:
        return None
        
    # Sort by distance to origin, take the N closest
    radii = np.hypot(points[:, 0], points[:, 1])
    indices = np.argsort(radii, kind='stable')[:N]
    return points[indices]


def _score(pts):
    """
    Compute the valid packing radius.
    Returns 0 if invalid.
    """
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
        
    # Check if inside unit circle
    dists_origin = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists_origin > 1.0):
        return 0.0
    r_cont = 1.0 - np.max(dists_origin)
    
    if N < 2:
        return r_cont
        
    # Pairwise distances
    # Using broadcasting for vectorization
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dmat = np.sqrt(dx ** 2 + dy ** 2)
    np.fill_diagonal(dmat, np.inf)
    min_pair = np.min(dmat)
    r_pair = min_pair / 2.0
    
    return min(r_cont, r_pair)


def _optimize(pts, max_iters=5000):
    """
    Force-based optimization using vectorized operations.
    """
    pts = pts.copy()
    n = len(pts)
    
    # Adaptive step size
    step_size = 0.01
    best_score = _score(pts)
    best_pts = pts.copy()
    
    # Pre-compute index pairs for efficiency if we were doing loops, 
    # but we are vectorizing the whole matrix.
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5.0:
            break
            
        # 1. Compute pairwise distances and forces
        # dx[i,j] = pts[i,0] - pts[j,0]
        dx = pts[:, 0:1] - pts[0:1, 0]
        dy = pts[:, 1:2] - pts[0:1, 1]
        
        # Squared distances
        d2 = dx ** 2 + dy ** 2
        
        # Avoid division by zero
        # We want to repel points that are close.
        # Force magnitude is proportional to 1/d^2, direction is d/d.
        # So vector force is d / d^3.
        # d^3 = (d^2)^(3/2)
        
        # To avoid numerical issues with very small d, add epsilon
        eps = 1e-6
        d2_safe = d2 + eps
        
        # Force components: F_ij = (p_i - p_j) / |p_i - p_j|^3
        # This is a standard Coulomb repulsion.
        inv_d3 = 1.0 / (d2_safe * np.sqrt(d2_safe))
        
        # Zero out diagonal
        inv_d3[np.diag_indices(n)] = 0.0
        
        # Sum forces for each point
        fx = np.sum(dx * inv_d3, axis=1)
        fy = np.sum(dy * inv_d3, axis=1)
        
        # 2. Boundary constraint
        # Push points inward if they are close to the boundary
        radii = np.hypot(pts[:, 0], pts[:, 1])
        # If radius > 0.8, apply inward force
        # Force is proportional to (r - 0.8) * direction_inward
        boundary_mask = radii > 0.8
        if np.any(boundary_mask):
            # Direction inward is -p / |p|
            # Magnitude k * (r - 0.8)
            k_boundary = 10.0
            # Only apply to those outside 0.8
            f_boundary_x = -k_boundary * (radii[boundary_mask] - 0.8) * pts[boundary_mask, 0] / radii[boundary_mask]
            f_boundary_y = -k_boundary * (radii[boundary_mask] - 0.8) * pts[boundary_mask, 1] / radii[boundary_mask]
            
            fx[boundary_mask] += f_boundary_x
            fy[boundary_mask] += f_boundary_y
            
        # 3. Update positions
        pts[:, 0] += step_size * fx
        pts[:, 1] += step_size * fy
        
        # 4. Clamp to unit circle
        radii = np.hypot(pts[:, 0], pts[:, 1])
        out_of_bounds = radii > 1.0
        if np.any(out_of_bounds):
            pts[out_of_bounds] /= radii[out_of_bounds, np.newaxis]
            
        # 5. Check score and adapt step size
        current_score = _score(pts)
        if current_score > best_score:
            best_score = current_score
            best_pts = pts.copy()
            # Increase step size slightly if improving
            step_size *= 1.05
            if step_size > 0.05:
                step_size = 0.05
        else:
            # Decrease step size if not improving
            step_size *= 0.95
            
        # Convergence check
        if it > 100 and step_size < 1e-5:
            break
            
    return best_pts


def construct():
    """
    Construct the packing of 50 circles in a unit circle.
    """
    best_pts = None
    best_score = 0.0
    
    # Try several initial spacings for the hex lattice
    # The optimal radius is ~0.125, so diameter ~0.25.
    # Hex lattice spacing should be around the diameter.
    spacings = [0.25, 0.26, 0.24, 0.27, 0.23, 0.28, 0.22]
    
    for spacing in spacings:
        if time.time() > _DEADLINE - 10.0:
            break
            
        pts_init = _hex_lattice_init(spacing)
        if pts_init is None:
            continue
            
        # Optimize
        pts_opt = _optimize(pts_init, max_iters=2000)
        score = _score(pts_opt)
        
        if score > best_score:
            best_score = score
            best_pts = pts_opt.copy()
            
        # If we found a very good solution, we can stop early
        # The theoretical max is bounded by area, but 0.1258 is known best.
        if best_score > 0.125:
            # Continue to try to improve slightly, but we have a strong candidate
            pass

    # If no valid points were found (should not happen with these spacings), fallback
    if best_pts is None:
        # Fallback to a simple grid or hex lattice without optimization
        pts_init = _hex_lattice_init(0.25)
        if pts_init is not None:
            best_pts = pts_init
        else:
            # Last resort: random
            best_pts = np.random.rand(N, 2) * 0.8 - 0.4
            
    return best_pts
