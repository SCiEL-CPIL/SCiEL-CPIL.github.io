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

def _hull_data(pts):
    """
    Compute convex hull and extract facet normals, radii, and vertex indices.
    Returns:
        radii: (F,) array of circumradii in radians for each facet
        normals: (F, 3) array of outward unit normals
        faces: (F, 3) array of vertex indices for each facet
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        valid = area > _EPS
        if not np.any(valid):
            return np.array([np.pi]), np.zeros((1, 3)), np.array([[0, 0, 0]])
        
        normals = cross[valid]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        
        return radii, normals, faces[valid]
    except Exception:
        return np.array([np.pi]), np.zeros((1, 3)), np.array([[0, 0, 0]])

def _hull_radius(pts):
    """Compute the exact covering radius."""
    radii, _, _ = _hull_data(pts)
    return np.max(radii)

def _lloyd_relax(pts, n_iter=50, deadline=None):
    """
    Perform Lloyd relaxation: for each point, compute the centroid of its
    Voronoi cell (approximated by the average of facet normals weighted by
    solid angle, or more simply, the pole of the worst adjacent facet).
    
    A simpler and more robust approach: for each point, collect all facets
    that contain it, compute the average of their normals (weighted by 1/radius),
    and move the point toward that average direction.
    """
    best_pts = pts.copy()
    best_r = _hull_radius(pts)
    
    for it in range(n_iter):
        if deadline and time.time() > deadline:
            break
            
        radii, normals, faces = _hull_data(pts)
        if len(radii) == 0:
            break
            
        curr_r = np.max(radii)
        
        # For each point, compute a "target" direction based on adjacent facets
        # We want to move each point to better cover its local region.
        # Strategy: for each facet, the pole is the worst point in that facet.
        # Each vertex of a facet should be pulled toward the pole.
        
        targets = np.zeros_like(pts)
        weights = np.zeros(N)
        
        for f in range(len(radii)):
            # Weight by how "bad" this facet is (larger radius = more important)
            w = radii[f] ** 2  # Emphasize worst facets
            
            for v in faces[f]:
                # Pull vertex v toward the pole of this facet
                targets[v] += w * normals[f]
                weights[v] += w
        
        # Normalize and move
        mask = weights > _EPS
        if np.any(mask):
            new_pts = pts.copy()
            new_pts[mask] = targets[mask] / (np.linalg.norm(targets[mask], axis=1, keepdims=True) + _EPS)
            # Blend with original to avoid large jumps
            new_pts[mask] = 0.5 * pts[mask] + 0.5 * new_pts[mask]
            new_pts[mask] /= (np.linalg.norm(new_pts[mask], axis=1, keepdims=True) + _EPS)
            
            new_r = _hull_radius(new_pts)
            if new_r < best_r - 1e-12:
                best_r = new_r
                best_pts = new_pts
                pts = new_pts
            else:
                break
        else:
            break
    
    return best_pts, best_r

def _pole_chase(pts, n_steps=100, deadline=None):
    """
    Deterministic pole-chasing: repeatedly find the worst facet, and move
    its three vertices toward the pole.
    """
    best_pts = pts.copy()
    best_r = _hull_radius(pts)
    step_size = 0.1
    
    for step in range(n_steps):
        if deadline and time.time() > deadline:
            break
            
        radii, normals, faces = _hull_data(pts)
        if len(radii) == 0:
            break
            
        curr_r = np.max(radii)
        if curr_r < best_r - 1e-12:
            best_r = curr_r
            best_pts = pts.copy()
            
        worst_idx = np.argmax(radii)
        v_indices = faces[worst_idx]
        pole = normals[worst_idx]
        
        cand = pts.copy()
        for v in v_indices:
            dir_vec = pole - pts[v]
            dir_norm = np.linalg.norm(dir_vec)
            if dir_norm > _EPS:
                dir_vec /= dir_norm
            else:
                continue
            cand[v] = pts[v] + step_size * dir_vec
            cand[v] /= (np.linalg.norm(cand[v]) + _EPS)
        
        cand_r = _hull_radius(cand)
        if cand_r < curr_r - 1e-14:
            pts = cand
        else:
            # Try smaller step
            step_size *= 0.8
            if step_size < 1e-6:
                break
    
    return best_pts, best_r

def _random_perturbation(pts, magnitude=0.1, seed=42):
    """Apply random perturbation to escape local minima."""
    rng = np.random.default_rng(seed)
    new_pts = pts + rng.standard_normal((N, 3)) * magnitude
    new_pts /= (np.linalg.norm(new_pts, axis=1, keepdims=True) + _EPS)
    return new_pts

def construct():
    deadline = time.time() + 145.0  # Safety margin for 150s limit
    
    best_pts = _initial_fib()
    best_r = _hull_radius(best_pts)
    
    # Phase 1: Initial Lloyd relaxation to get a reasonable starting point
    if time.time() < deadline - 60.0:
        best_pts, best_r = _lloyd_relax(best_pts, n_iter=30, deadline=deadline - 60.0)
    
    # Phase 2: Multi-restart with pole-chasing and Lloyd relaxation
    n_restarts = 8
    time_per_restart = (deadline - 30.0) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        # Create a perturbed version of the best found so far
        if restart == 0:
            pts = best_pts.copy()
        else:
            seed = 100 + restart * 7
            magnitude = 0.15 + 0.05 * (restart % 3)
            pts = _random_perturbation(best_pts, magnitude=magnitude, seed=seed)
        
        # Allocate time for this restart
        restart_deadline = min(deadline - 30.0, time.time() + time_per_restart)
        
        # Sub-phase A: Lloyd relaxation
        if time.time() < restart_deadline - 5.0:
            pts, r = _lloyd_relax(pts, n_iter=20, deadline=restart_deadline - 5.0)
        
        # Sub-phase B: Pole chasing
        if time.time() < restart_deadline:
            pts, r = _pole_chase(pts, n_steps=50, deadline=restart_deadline)
        
        # Sub-phase C: Final Lloyd polish
        if time.time() < deadline - 25.0:
            pts, r = _lloyd_relax(pts, n_iter=15, deadline=deadline - 25.0)
        
        if r < best_r - 1e-12:
            best_r = r
            best_pts = pts
    
    # Phase 3: Final intensive polishing
    if time.time() < deadline - 5.0:
        best_pts, best_r = _pole_chase(best_pts, n_steps=200, deadline=deadline - 5.0)
        if time.time() < deadline - 2.0:
            best_pts, best_r = _lloyd_relax(best_pts, n_iter=30, deadline=deadline - 2.0)
    
    # Final normalization and validity check
    best_pts = np.array(best_pts, dtype=float)
    best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
