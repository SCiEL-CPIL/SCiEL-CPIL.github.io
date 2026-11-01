import time
import numpy as np
from scipy.spatial import ConvexHull

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

def _energy(pts, grid_pts):
    """
    Compute the sum of squared angular distances to the grid points.
    This is a smooth surrogate for the covering radius.
    """
    # dots[i, j] = pts[i] @ grid_pts[j]
    dots = pts @ grid_pts.T
    # For each grid point j, find the index of the nearest point i
    nearest_idx = np.argmax(dots, axis=0)
    
    # Compute angular distance for each grid point
    # cos(theta) = pts[nearest_idx[j]] @ grid_pts[j]
    nearest_pts = pts[nearest_idx]
    cos_theta = np.sum(nearest_pts * grid_pts, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    ang_dists = np.arccos(cos_theta)
    
    return np.sum(ang_dists ** 2)

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
    G = 5000
    i = np.arange(G) + 0.5
    z = 1.0 - 2.0 * i / G
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    grid_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Multi-restart stochastic local search (Simulated Annealing)
    best_pts = _initial_fib()
    best_energy = _energy(best_pts, grid_pts)
    
    # SA parameters
    T0 = 0.05
    T_end = 0.001
    alpha = 0.95
    max_iters = 500
    
    n_restarts = 10
    time_per_restart = (deadline - 5.0) / (n_restarts + 1)  # Reserve 5s for final polish
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize
        if restart == 0:
            pts = _initial_fib()
        else:
            pts = _random_init(rng)
        
        curr_energy = _energy(pts, grid_pts)
        if curr_energy < best_energy:
            best_energy = curr_energy
            best_pts = pts.copy()
        
        T = T0
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            # Pick a random point to move
            idx = rng.integers(N)
            
            # Generate a random perturbation
            # Small random step on the sphere
            step_size = 0.05 * (T / T0) + 0.01
            noise = rng.standard_normal(3)
            noise /= (np.linalg.norm(noise) + _EPS)
            
            cand = pts.copy()
            cand[idx] = cand[idx] + step_size * noise
            cand[idx] /= (np.linalg.norm(cand[idx]) + _EPS)
            
            cand_energy = _energy(cand, grid_pts)
            delta_E = cand_energy - curr_energy
            
            # Accept if better or with probability exp(-delta_E / T)
            if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                pts = cand
                curr_energy = cand_energy
                if curr_energy < best_energy:
                    best_energy = curr_energy
                    best_pts = pts.copy()
            
            # Cool down
            T *= alpha
            if T < T_end:
                break
    
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
