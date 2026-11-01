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

def _surrogate_objective_and_grad(x, grid_pts):
    """
    Compute the smooth surrogate objective (sum of squared angular distances)
    and its gradient for L-BFGS-B optimization.
    
    x: flattened array of shape (N*3,)
    grid_pts: array of shape (G, 3)
    
    Returns:
        objective: sum of squared angular distances
        gradient: array of shape (N*3,)
    """
    pts = x.reshape(N, 3)
    G = len(grid_pts)
    
    # Compute dot products: dots[i, j] = pts[i] @ grid_pts[j]
    dots = pts @ grid_pts.T  # (N, G)
    
    # For each grid point j, find the nearest point i
    nearest_idx = np.argmax(dots, axis=0)  # (G,)
    
    # Compute angular distance for each grid point
    nearest_pts = pts[nearest_idx]  # (G, 3)
    cos_theta = np.sum(nearest_pts * grid_pts, axis=1)  # (G,)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    ang_dists = np.arccos(cos_theta)  # (G,)
    
    # Objective: sum of squared angular distances
    objective = np.sum(ang_dists ** 2)
    
    # Gradient computation:
    # d/dx [arccos(pts[i] @ grid_pts[j])] = -grid_pts[j] / sqrt(1 - (pts[i] @ grid_pts[j])^2)
    # But we need to account for the nearest-neighbor selection.
    # For points that are the nearest to some grid points, the gradient is non-zero.
    
    # Create a sparse representation of the gradient
    grad = np.zeros_like(pts)
    
    # For each grid point j, the contribution to the gradient of pts[i] (where i = nearest_idx[j])
    # is: -2 * ang_dists[j] * grid_pts[j] / sin(ang_dists[j])
    # (since d/dtheta [theta^2] = 2*theta, and dtheta/dx = -grid_pts[j] / sin(theta))
    
    sin_theta = np.sin(ang_dists)
    # Avoid division by zero
    valid_sin = sin_theta > _EPS
    if np.any(valid_sin):
        weights = -2.0 * ang_dists[valid_sin] / sin_theta[valid_sin]
        # Accumulate gradients
        for j in np.where(valid_sin)[0]:
            i = nearest_idx[j]
            grad[i] += weights[j] * grid_pts[j]
    
    return objective, grad.flatten()

def _lloyd_relax(pts, grid_pts, max_iters=50, tol=1e-6):
    """
    Perform Lloyd relaxation: move each point to the centroid of its Voronoi cell
    (approximated by the mean of grid points assigned to it).
    """
    for _ in range(max_iters):
        # Compute dot products
        dots = pts @ grid_pts.T  # (N, G)
        
        # Assign each grid point to nearest point
        nearest_idx = np.argmax(dots, axis=0)  # (G,)
        
        # Compute new positions: centroid of assigned grid points
        new_pts = np.zeros_like(pts)
        counts = np.zeros(N)
        
        for j in range(len(grid_pts)):
            i = nearest_idx[j]
            new_pts[i] += grid_pts[j]
            counts[i] += 1
        
        # Avoid division by zero for points with no assigned grid points
        valid_counts = counts > 0
        new_pts[valid_counts] /= counts[valid_counts, np.newaxis]
        
        # Normalize to unit sphere
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        new_pts /= (norms + _EPS)
        
        # Check convergence
        diff = np.linalg.norm(new_pts - pts)
        pts = new_pts
        if diff < tol:
            break
    
    return pts

def _local_polish(pts, deadline):
    """
    Perform high-iteration count of facet-based local optimization.
    Moves the worst facet's vertices towards its pole.
    """
    best_pts = pts.copy()
    best_r = _compute_hull_radius(pts)
    
    # Adaptive step size
    step = 0.02
    no_improve_count = 0
    
    while time.time() < deadline:
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
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 20:
                    step *= 0.95
                    no_improve_count = 0
                    if step < 1e-5:
                        break
            
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
                    dir_vec = 0.0
                cand[v] = pts[v] + step * dir_vec
                cand[v] /= (np.linalg.norm(cand[v]) + _EPS)
                
            # Check improvement
            cand_r = _compute_hull_radius(cand)
            if cand_r < curr_r:
                pts = cand
            else:
                # Try smaller step if stuck
                step *= 0.8
                
        except Exception:
            break
            
    return best_pts, best_r

def construct():
    deadline = time.time() + 145.0  # Safety margin
    rng = np.random.default_rng(42)
    
    # Create a dense spherical grid for the energy function
    G = 10000
    i = np.arange(G) + 0.5
    z = 1.0 - 2.0 * i / G
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    grid_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Multi-restart with L-BFGS-B optimization
    best_pts = _initial_fib()
    best_energy = _surrogate_objective_and_grad(best_pts.flatten(), grid_pts)[0]
    
    n_restarts = 5
    time_per_restart = (deadline - 10.0) / (n_restarts + 1)  # Reserve 10s for final polish
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Initialize
        if restart == 0:
            pts = _initial_fib()
        else:
            pts = _random_init(rng)
        
        # First, do Lloyd relaxation to get a good starting point
        pts = _lloyd_relax(pts, grid_pts, max_iters=20, tol=1e-5)
        
        curr_energy = _surrogate_objective_and_grad(pts.flatten(), grid_pts)[0]
        if curr_energy < best_energy:
            best_energy = curr_energy
            best_pts = pts.copy()
        
        # Now use L-BFGS-B to refine
        x0 = pts.flatten()
        try:
            result = minimize(
                lambda x: _surrogate_objective_and_grad(x, grid_pts),
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 100, 'gtol': 1e-4, 'maxfun': 200}
            )
            
            if result.success or result.fun < curr_energy:
                # Unflatten and normalize
                new_pts = result.x.reshape(N, 3)
                new_pts /= (np.linalg.norm(new_pts, axis=1, keepdims=True) + _EPS)
                
                new_energy = _surrogate_objective_and_grad(new_pts.flatten(), grid_pts)[0]
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pts = new_pts.copy()
        except Exception:
            pass
    
    # Final local polish using exact hull metric
    if time.time() < deadline - 2.0:
        best_pts, best_r = _local_polish(best_pts, deadline)
        
    # Final normalization
    best_pts = np.array(best_pts, dtype=float)
    best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    
    # Validity check
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
