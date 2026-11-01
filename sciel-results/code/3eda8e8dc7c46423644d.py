import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130
_EPS = 1e-12

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_init(rng):
    """Generate random initial points."""
    pts = rng.standard_normal((N, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _compute_hull_radius(pts):
    """
    Compute the exact covering radius using the convex hull method.
    Returns the radius in radians.
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        M = len(faces)
        
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        # Vectors for the triangle edges
        AB = B - A
        AC = C - A
        
        # Area of triangle: 0.5 * || AB x AC ||
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        # Handle degenerate facets
        valid = area > _EPS
        if not np.any(valid):
            return np.pi
            
        # Normal vector for each facet
        normals = cross[valid]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        
        # Ensure normals point outward (away from origin)
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        # Angular circumradius: distance from pole to any vertex (e.g., A)
        # cos(theta) = P @ A
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        ang_radii = np.arccos(cos_theta)
        
        return np.max(ang_radii)
    except Exception:
        return np.pi

def _lloyd_relax(pts, n_iters=50, deadline=None):
    """
    Lloyd's relaxation: iteratively move each point to the centroid of its Voronoi cell.
    We approximate the Voronoi cell using a dense grid.
    """
    G = 2000
    i = np.arange(G) + 0.5
    z = 1.0 - 2.0 * i / G
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    grid_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    for it in range(n_iters):
        if deadline and time.time() > deadline:
            break
            
        # Assign each grid point to nearest configuration point
        dots = grid_pts @ pts.T
        nearest = np.argmax(dots, axis=1)
        
        # Compute centroid of each cell
        new_pts = np.zeros_like(pts)
        counts = np.zeros(N)
        
        # Use bincount to accumulate sums
        # For each dimension
        for d in range(3):
            new_pts[:, d] = np.bincount(nearest, weights=grid_pts[:, d], minlength=N)
            counts = np.bincount(nearest, minlength=N)
        
        # Normalize: centroid / count, then normalize to unit sphere
        valid = counts > 0
        for i in range(N):
            if valid[i]:
                centroid = new_pts[i] / counts[i]
                norm = np.linalg.norm(centroid)
                if norm > _EPS:
                    pts[i] = centroid / norm
    
    return pts

def _subgradient_descent(pts, n_iters=100, deadline=None, step_init=0.05):
    """
    Subgradient descent on the max circumradius.
    Move the vertices of the worst facet towards its circumcenter (pole).
    """
    best_pts = pts.copy()
    best_r = _compute_hull_radius(pts)
    step = step_init
    
    for it in range(n_iters):
        if deadline and time.time() > deadline:
            break
            
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            M = len(faces)
            
            A = pts[faces[:, 0]]
            B = pts[faces[:, 1]]
            C = pts[faces[:, 2]]
            
            AB = B - A
            AC = C - A
            cross = np.cross(AB, AC)
            area = 0.5 * np.linalg.norm(cross, axis=1)
            valid = area > _EPS
            if not np.any(valid):
                break
                
            normals = cross[valid]
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            centroids = (A[valid] + B[valid] + C[valid]) / 3.0
            signs = np.sum(normals * centroids, axis=1, keepdims=True)
            normals = normals * np.sign(signs)
            normals = normals / (norms + _EPS)
            
            cos_theta = np.sum(normals * A[valid], axis=1)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            ang_radii = np.arccos(cos_theta)
            
            curr_r = np.max(ang_radii)
            
            if curr_r < best_r:
                best_r = curr_r
                best_pts = pts.copy()
            
            # Find worst facet
            worst_idx = np.argmax(ang_radii)
            v_indices = faces[valid][worst_idx]
            pole = normals[worst_idx]
            
            cand = pts.copy()
            for v in v_indices:
                dir_vec = pole - pts[v]
                dir_norm = np.linalg.norm(dir_vec)
                if dir_norm > _EPS:
                    dir_vec /= dir_norm
                else:
                    continue
                cand[v] = pts[v] + step * dir_vec
                cand[v] /= (np.linalg.norm(cand[v]) + _EPS)
                
            cand_r = _compute_hull_radius(cand)
            if cand_r < curr_r:
                pts = cand
                step *= 1.0  # Keep step
            else:
                step *= 0.7  # Reduce step
                
        except Exception:
            break
    
    return best_pts, best_r

def _surrogate_objective_and_grad(x, grid_pts):
    """
    Compute smooth surrogate (sum of squared angular distances) and its gradient.
    x: flattened array of shape (N*3,)
    grid_pts: array of shape (G, 3)
    """
    pts = x.reshape(N, 3)
    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts_norm = pts / (norms + _EPS)
    
    # dots[i, j] = pts_norm[i] @ grid_pts[j]
    dots = pts_norm @ grid_pts.T
    
    # For each grid point j, find the index of the nearest point i
    nearest_idx = np.argmax(dots, axis=0)
    
    # Compute angular distance for each grid point
    nearest_pts = pts_norm[nearest_idx]
    cos_theta = np.sum(nearest_pts * grid_pts, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    ang_dists = np.arccos(cos_theta)
    
    # Objective: sum of squared angular distances
    obj = np.sum(ang_dists ** 2)
    
    # Gradient: d/dx sum_j (arccos(cos_theta_j))^2
    # Let f_j = arccos(c_j), where c_j = p_{n_j} . g_j
    # df_j/dp_{n_j} = -g_j / sqrt(1 - c_j^2)
    # d(obj)/dx_i = sum_{j: n_j == i} 2 * f_j * df_j/dp_i
    
    grad_pts = np.zeros_like(pts)
    
    # Compute sqrt(1 - c^2) for valid entries
    sin_theta = np.sqrt(np.clip(1.0 - cos_theta ** 2, _EPS, None))
    
    # For each grid point j, the gradient contribution to point n_j is:
    # 2 * ang_dists[j] * (-grid_pts[j] / sin_theta[j])
    # = -2 * ang_dists[j] / sin_theta[j] * grid_pts[j]
    
    weights = -2.0 * ang_dists / sin_theta
    # weights shape: (G,)
    # For each j, add weights[j] * grid_pts[j] to grad_pts[nearest_idx[j]]
    
    for d in range(3):
        grad_pts[:, d] = np.bincount(nearest_idx, weights=weights * grid_pts[:, d], minlength=N)
    
    # Chain rule: dx -> d(pts) / norms
    # pts = x / norms, so d(pts_i)/dx_i = (I - pts_i pts_i^T) / norms_i
    # grad_x_i = grad_pts_i / norms_i - (grad_pts_i . pts_i / norms_i) * pts_i / norms_i
    #           = (grad_pts_i - (grad_pts_i . pts_norm_i) * pts_norm_i) / norms_i
    
    grad = np.zeros_like(x)
    for i in range(N):
        p_norm = pts_norm[i]
        g_i = grad_pts[i]
        # Project g_i onto tangent space at p_norm_i
        proj = g_i - np.dot(g_i, p_norm) * p_norm
        grad[i*3:(i+1)*3] = proj / (norms[i][0] + _EPS)
    
    return obj, grad

def _hull_radii_and_poles(pts):
    """Compute all facet circumradii and poles."""
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        M = len(faces)
        
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        valid = area > _EPS
        
        if not np.any(valid):
            return np.array([]), np.array([]), faces
        
        normals = cross[valid]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        ang_radii = np.arccos(cos_theta)
        
        return ang_radii, normals, faces[valid]
    except Exception:
        return np.array([]), np.array([]), np.array([])

def construct():
    deadline = time.time() + 145.0  # Safety margin
    rng = np.random.default_rng(42)
    
    # Create a dense spherical grid for the energy function
    G = 5000
    i = np.arange(G) + 0.5
    z = 1.0 - 2.0 * i / G
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    grid_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = _initial_fib()
    best_r = _compute_hull_radius(best_pts)
    
    # Strategy: Multiple restarts with L-BFGS-B on the smooth surrogate,
    # followed by exact hull-based subgradient polishing.
    
    n_restarts = 8
    time_per_restart = (deadline - 15.0) / (n_restarts + 1)  # Reserve 15s for final polish
    
    for restart in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
            
        # Initialize
        if restart == 0:
            pts = _initial_fib()
        elif restart == 1:
            pts = _lloyd_relax(_initial_fib(), n_iters=30, deadline=deadline - 15.0)
        else:
            pts = _random_init(rng)
        
        # First, do a quick Lloyd relaxation to get a reasonable starting point
        if restart > 0:
            pts = _lloyd_relax(pts, n_iters=20, deadline=deadline - 15.0)
        
        # Check if already good
        r_curr = _compute_hull_radius(pts)
        if r_curr < best_r:
            best_r = r_curr
            best_pts = pts.copy()
        
        # Use L-BFGS-B on the smooth surrogate
        x0 = pts.flatten()
        
        try:
            res = minimize(
                _surrogate_objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 50, 'gtol': 1e-6, 'ftol': 1e-8}
            )
            
            pts_opt = res.x.reshape(N, 3)
            pts_opt /= (np.linalg.norm(pts_opt, axis=1, keepdims=True) + _EPS)
            
            r_opt = _compute_hull_radius(pts_opt)
            if r_opt < best_r:
                best_r = r_opt
                best_pts = pts_opt.copy()
                
        except Exception:
            pass
    
    # Final local polish using exact hull metric
    if time.time() < deadline - 2.0:
        best_pts, best_r = _subgradient_descent(best_pts, n_iters=200, deadline=deadline - 1.0, step_init=0.03)
        
    # Final normalization
    best_pts = np.array(best_pts, dtype=float)
    best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    
    # Validity check
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
