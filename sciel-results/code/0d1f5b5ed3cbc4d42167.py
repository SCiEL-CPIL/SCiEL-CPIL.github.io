"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Precompute a high-resolution icosahedral grid for fast covering-radius evaluation
def _make_grid(n_subdiv=40):
    """Create a high-resolution grid on the sphere using icosahedral subdivision."""
    # Start with icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Faces of the icosahedron
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]
    
    # Subdivide each face
    def subdivide_face(a, b, c, level, cache):
        if level == 0:
            return [a, b, c]
        ab = cache.get((a, b))
        if ab is None:
            ab = (a + b) / np.linalg.norm(a + b)
            cache[(a, b)] = ab
        bc = cache.get((b, c))
        if bc is None:
            bc = (b + c) / np.linalg.norm(b + c)
            cache[(b, c)] = bc
        ca = cache.get((c, a))
        if ca is None:
            ca = (c + a) / np.linalg.norm(c + a)
            cache[(c, a)] = ca
        return (subdivide_face(a, ab, ca, level-1, cache) +
                subdivide_face(ab, b, bc, level-1, cache) +
                subdivide_face(ca, bc, c, level-1, cache) +
                subdivide_face(ab, bc, ca, level-1, cache))
    
    cache = {}
    all_points = []
    for f in faces:
        a, b, c = verts[f[0]], verts[f[1]], verts[f[2]]
        all_points.extend(subdivide_face(a, b, c, n_subdiv, cache))
    
    # Deduplicate
    all_points = np.array(all_points)
    all_points /= np.linalg.norm(all_points, axis=1, keepdims=True)
    unique, idx = np.unique(np.round(all_points, 10), axis=0, return_index=True)
    return all_points[idx]

# Precompute grid once at module level
_GRID = _make_grid(n_subdiv=30)  # ~30^2 * 20 = ~18000 points, good resolution

def _covering_radius(points):
    """Compute covering radius using the precomputed grid."""
    # points: (N, 3) unit vectors
    # _GRID: (M, 3) unit vectors
    # For each grid point, find min angular distance to any point
    # Use dot product: cos(angle) = dot product
    # max angle = arccos(min dot product)
    
    # Compute all dot products: (M, N)
    dots = _GRID @ points.T
    # For each grid point, find the max dot product (closest point)
    max_dots = np.max(dots, axis=1)
    # Clip for numerical safety
    max_dots = np.clip(max_dots, -1.0, 1.0)
    # The covering radius is the max over grid points of arccos(max_dot)
    # We want to minimize this
    angles = np.arccos(max_dots)
    return np.max(angles)

def _covering_radius_grad(points):
    """Compute covering radius and gradient using the precomputed grid."""
    # points: (N, 3) unit vectors
    dots = _GRID @ points.T  # (M, N)
    max_dots = np.max(dots, axis=1)
    max_idx = np.argmax(dots, axis=1)
    
    # For each grid point, the closest point index
    # Gradient with respect to points[i]
    # d/dpoints[i] arccos(dot) = -dot / (sqrt(1-dot^2)) * (d dot / d points[i])
    # d dot / d points[i] = grid_point (for the i-th component)
    
    # Only the max-index point contributes to the gradient
    # For grid point j, the closest point is max_idx[j]
    # Contribution to gradient of points[max_idx[j]]:
    #   -max_dots[j] / sqrt(1 - max_dots[j]^2) * _GRID[j]
    
    # Clip for safety
    max_dots_clipped = np.clip(max_dots, -0.999999, 0.999999)
    denom = np.sqrt(1.0 - max_dots_clipped**2)
    denom = np.where(denom < 1e-10, 1e-10, denom)
    
    # Weight for each grid point
    weights = -max_dots_clipped / denom
    
    # Accumulate gradients
    grad = np.zeros_like(points)
    # For each grid point j, add weights[j] * _GRID[j] to grad[max_idx[j]]
    np.add.at(grad, max_idx, weights[:, None] * _GRID)
    
    angles = np.arccos(max_dots_clipped)
    cr = np.max(angles)
    
    return cr, grad

def _project_to_sphere(points):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1e-10, norms)
    return points / norms

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # Initial configurations
    configs = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, None))
    configs.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
    
    # 2. Random
    rng = np.random.RandomState(42)
    for _ in range(3):
        pts = rng.randn(N, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        configs.append(pts)
    
    # 3. Icosahedral-based
    base = _GRID
    # Select N points from the grid
    idx = np.random.RandomState(123).choice(len(base), N, replace=False)
    configs.append(base[idx])
    
    best_points = configs[0].copy()
    best_cr = _covering_radius(best_points)
    
    for init_idx, init_pts in enumerate(configs):
        if time.time() > deadline:
            break
        
        # Optimize using L-BFGS-B
        # We parameterize points in R^3 and project to sphere after each step
        # Use a smooth objective for gradient-based optimization
        
        def objective(x_flat):
            pts = _project_to_sphere(x_flat.reshape(N, 3))
            return _covering_radius(pts)
        
        def objective_grad(x_flat):
            pts = _project_to_sphere(x_flat.reshape(N, 3))
            cr, grad = _covering_radius_grad(pts)
            # Project gradient back: for sphere constraint, we need to account for projection
            # Simple approach: just use the gradient as-is, the projection handles the constraint
            return grad.flatten()
        
        x0 = init_pts.flatten()
        
        try:
            res = minimize(
                objective, x0,
                jac=objective_grad,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            opt_pts = _project_to_sphere(res.x.reshape(N, 3))
            cr = _covering_radius(opt_pts)
            
            if cr < best_cr:
                best_cr = cr
                best_points = opt_pts.copy()
        except Exception:
            pass
    
    # Final polish: local refinement
    if time.time() < deadline - 10:
        def objective(x_flat):
            pts = _project_to_sphere(x_flat.reshape(N, 3))
            return _covering_radius(pts)
        
        def objective_grad(x_flat):
            pts = _project_to_sphere(x_flat.reshape(N, 3))
            cr, grad = _covering_radius_grad(pts)
            return grad.flatten()
        
        x0 = best_points.flatten()
        try:
            res = minimize(
                objective, x0,
                jac=objective_grad,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            opt_pts = _project_to_sphere(res.x.reshape(N, 3))
            cr = _covering_radius(opt_pts)
            if cr < best_cr:
                best_cr = cr
                best_points = opt_pts.copy()
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
