import time
import numpy as np
from scipy.spatial import ConvexHull

N = 130
_EPS = 1e-12
_DEADLINE = 145.0

def _initial_fib():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_hull_data(pts):
    """
    Compute per-facet covering radius and outward pole direction.
    Returns (radii, normals, faces, valid_mask).
    """
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
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        normals = cross[valid].copy()
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        
        return radii, normals, faces[valid], valid
    except Exception:
        return np.array([]), np.array([]), np.array([]), np.array([])

def _hull_radius(pts):
    """
    Compute the exact covering radius using the convex hull.
    Returns the maximum angular distance from the sphere to the nearest point.
    """
    radii, _, _, _ = _compute_hull_data(pts)
    if len(radii) == 0:
        return np.pi
    return radii.max()

def _lloyd_relax(pts, n_iter=50, deadline=None):
    """
    Lloyd relaxation: each point moves to the centroid of its Voronoi cell (approximated by hull facets).
    We approximate the Voronoi cell on the sphere by the union of spherical triangles formed by the
    adjacent hull facets. For each point, we collect all hull facets incident to it, compute the
    spherical centroid (normalized mean of the vertices and the pole), and move the point there.
    """
    pts = pts.astype(float).copy()
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        radii, normals, faces, valid = _compute_hull_data(pts)
        if len(radii) == 0:
            break
        
        M = len(faces)
        new_pts = pts.copy()
        
        # For each point, accumulate the centroid of its Voronoi cell
        # Approximation: for each point, take the average of the centroids of all facets incident to it,
        # and also include the poles of those facets, to better approximate the spherical centroid.
        cell_sum = np.zeros((N, 3))
        cell_count = np.zeros(N)
        
        for f in range(M):
            # Facet f has vertices faces[f, 0], faces[f, 1], faces[f, 2]
            # Its centroid on the sphere is approximately the normalized mean of the three vertices
            # and the pole (for better spherical approximation)
            centroid = (pts[faces[f, 0]] + pts[faces[f, 1]] + pts[faces[f, 2]]) / 3.0
            centroid = centroid + normals[f]  # Add pole for better spherical centroid
            centroid = centroid / (np.linalg.norm(centroid) + _EPS)
            
            for v in faces[f]:
                cell_sum[v] += centroid
                cell_count[v] += 1
        
        # Update points that have at least one incident facet
        valid_pts = cell_count > 0
        new_pts[valid_pts] = cell_sum[valid_pts] / cell_count[valid_pts, np.newaxis]
        new_pts[valid_pts] = new_pts[valid_pts] / (np.linalg.norm(new_pts[valid_pts], axis=1, keepdims=True) + _EPS)
        
        # Check for convergence
        diff = np.linalg.norm(new_pts - pts, axis=1)
        pts = new_pts
        if np.max(diff) < 1e-10:
            break
    
    return pts

def _subgradient_descent(pts, deadline, max_iter=2000):
    """
    Subgradient-like descent: move the worst facet's vertices towards the pole.
    This is a simple but effective method for reducing the covering radius.
    """
    pts = pts.astype(float).copy()
    best_r = _hull_radius(pts)
    best_pts = pts.copy()
    
    step = 0.01
    no_improve = 0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        radii, normals, faces, valid = _compute_hull_data(pts)
        if len(radii) == 0:
            break
        
        curr_r = radii.max()
        
        if curr_r < best_r - 1e-10:
            best_r = curr_r
            best_pts = pts.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 20:
                step *= 0.9
                no_improve = 0
                if step < 1e-6:
                    break
        
        worst_idx = np.argmax(radii)
        worst_face = faces[worst_idx]
        pole = normals[worst_idx]
        
        cand = pts.copy()
        for v in worst_face:
            # Move point v towards the pole in the tangent plane
            tangent_dir = pole - pts[v]
            tangent_dir = tangent_dir - np.dot(tangent_dir, pts[v]) * pts[v]
            norm_t = np.linalg.norm(tangent_dir)
            if norm_t > _EPS:
                tangent_dir /= norm_t
            else:
                continue
            cand[v] = pts[v] + step * tangent_dir
            cand[v] = cand[v] / (np.linalg.norm(cand[v]) + _EPS)
        
        new_r = _hull_radius(cand)
        if new_r < curr_r:
            pts = cand
        else:
            step *= 0.95
    
    return best_pts, best_r

def _random_perturbation(pts, rng, strength=0.01):
    """
    Randomly perturb a subset of points to escape local minima.
    """
    cand = pts.copy()
    n_perturb = max(1, N // 10)
    indices = rng.choice(N, size=n_perturb, replace=False)
    for i in indices:
        # Random tangent direction
        random_vec = rng.normal(size=3)
        tangent_dir = random_vec - np.dot(random_vec, cand[i]) * cand[i]
        norm_t = np.linalg.norm(tangent_dir)
        if norm_t > _EPS:
            tangent_dir /= norm_t
        else:
            continue
        cand[i] = cand[i] + strength * tangent_dir
        cand[i] = cand[i] / (np.linalg.norm(cand[i]) + _EPS)
    return cand

def _multi_restart_search(initial_pts, deadline):
    """
    Multi-restart search: try multiple initializations (Fibonacci, random) and
    alternate between Lloyd relaxation and subgradient descent.
    """
    best_pts = initial_pts.copy()
    best_r = _hull_radius(best_pts)
    
    rng = np.random.default_rng(42)
    
    # Create multiple initializations
    initializations = [initial_pts.copy()]
    for _ in range(3):
        random_pts = _random_perturbation(initial_pts, rng, strength=0.1)
        initializations.append(random_pts)
    
    # Also try a fully random initialization
    random_init = np.zeros((N, 3))
    for i in range(N):
        theta = rng.uniform(0, 2 * np.pi)
        z = rng.uniform(-1, 1)
        r = np.sqrt(np.clip(1 - z**2, 0, None))
        random_init[i] = np.array([r * np.cos(theta), r * np.sin(theta), z])
    initializations.append(random_init)
    
    for init in initializations:
        if time.time() > deadline - 5.0:
            break
        
        # Lloyd relaxation
        pts_lloyd = _lloyd_relax(init, n_iter=30, deadline=deadline - 5.0)
        r_lloyd = _hull_radius(pts_lloyd)
        
        if r_lloyd < best_r:
            best_r = r_lloyd
            best_pts = pts_lloyd.copy()
        
        # Subgradient descent from Lloyd result
        if time.time() < deadline - 10.0:
            pts_sub, r_sub = _subgradient_descent(pts_lloyd, deadline - 5.0, max_iter=1000)
            if r_sub < best_r:
                best_r = r_sub
                best_pts = pts_sub.copy()
        
        # Subgradient descent from original init
        if time.time() < deadline - 10.0:
            pts_sub2, r_sub2 = _subgradient_descent(init, deadline - 5.0, max_iter=1000)
            if r_sub2 < best_r:
                best_r = r_sub2
                best_pts = pts_sub2.copy()
    
    return best_pts, best_r

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # 1. Initial point: Fibonacci spiral
    pts_init = _initial_fib()
    
    # 2. Multi-restart search
    best_pts, best_r = _multi_restart_search(pts_init, deadline)
    
    # 3. Final polish with subgradient descent
    if time.time() < deadline - 2.0:
        best_pts, best_r = _subgradient_descent(best_pts, deadline, max_iter=500)
    
    # Final normalization
    best_pts = np.array(best_pts, dtype=float)
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / (norms + _EPS)
    
    # Validity check
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
    
    return best_pts
