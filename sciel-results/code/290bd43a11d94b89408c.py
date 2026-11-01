"""
Vectorized Facet-Based Covering Optimization.

Strategy:
1. Compute the convex hull of the points.
2. Vectorize the calculation of the circumradius for every facet (triangle) of the hull.
3. Identify the facet with the largest circumradius (the bottleneck for the covering radius).
4. Compute the pole (normal) of that facet.
5. Move the three vertices of that facet towards the pole to reduce the circumradius.
6. Repeat until time limit.

This avoids the expensive Python loop over facets and repeated small matrix solves found in the seed program,
allowing for significantly more optimization steps within the time budget.
"""
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

def _vectorized_circumradii_and_poles(pts, hull):
    """
    Compute circumradii and poles for all hull facets in a vectorized manner.
    Returns:
        radii: array of shape (M,) with circumradius (radians) for each facet.
        poles: array of shape (M, 3) with normalized pole vectors for each facet.
        faces: array of shape (M, 3) with vertex indices for each facet.
    """
    faces = hull.simplices
    M = len(faces)
    
    # Get coordinates of vertices for each facet
    # faces: (M, 3) indices
    A = pts[faces[:, 0]]  # (M, 3)
    B = pts[faces[:, 1]]  # (M, 3)
    C = pts[faces[:, 2]]  # (M, 3)
    
    # Vectors for the triangle edges
    AB = B - A
    AC = C - A
    BC = C - B
    
    # Side lengths (Euclidean chord lengths)
    len_AB = np.linalg.norm(AB, axis=1)
    len_AC = np.linalg.norm(AC, axis=1)
    len_BC = np.linalg.norm(BC, axis=1)
    
    # Area of triangle: 0.5 * || AB x AC ||
    cross = np.cross(AB, AC)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    
    # Circumradius R = (a*b*c) / (4*Area)
    # a, b, c are side lengths
    prod_sides = len_AB * len_AC * len_BC
    denom = 4.0 * area
    
    # Handle degenerate facets (area ~ 0)
    valid = denom > _EPS
    radii = np.full(M, np.inf)
    radii[valid] = prod_sides[valid] / denom[valid]
    
    # Convert Euclidean circumradius to angular circumradius
    # For a unit sphere, the chord length c = 2 * sin(theta/2)
    # The circumradius of the triangle in 3D space is R_3d.
    # The angular radius theta is the distance from the pole to the vertices.
    # The pole P is the normal to the facet. The distance from P to A is the angular radius.
    # Actually, the "circumradius" in the context of covering is the angular distance.
    # The pole P is the unit normal to the facet.
    # The angular radius is arccos(P @ A).
    # Let's compute the poles first.
    
    # Normal vector for each facet
    normals = cross
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    # Ensure normals point outward (away from origin)
    # The centroid of the facet is (A+B+C)/3. If normal @ centroid > 0, it's outward.
    centroids = (A + B + C) / 3.0
    signs = np.sum(normals * centroids, axis=1, keepdims=True)
    normals = normals * np.sign(signs)
    normals = normals / (norms + _EPS)
    
    # Angular circumradius: distance from pole to any vertex (e.g., A)
    # cos(theta) = P @ A
    cos_theta = np.sum(normals * A, axis=1)
    # Clip to avoid numerical issues in arccos
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    ang_radii = np.arccos(cos_theta)
    
    # Only valid facets have finite radii
    ang_radii[~valid] = np.inf
    
    return ang_radii, normals, faces

def _improve_vectorized(pts, deadline, rng, max_iters=5000, step0=0.05, decay=0.9999):
    """
    Optimize the configuration by moving vertices of the worst facet towards its pole.
    """
    pts = np.array(pts, dtype=float)
    best = pts.copy()
    
    # Initial hull and radius
    try:
        hull = ConvexHull(pts)
        radii, poles, faces = _vectorized_circumradii_and_poles(pts, hull)
        best_r = np.max(radii)
    except Exception:
        return best, np.pi

    if not np.isfinite(best_r):
        best_r = np.pi

    step = step0
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        # Recompute hull and radii
        try:
            hull = ConvexHull(pts)
            radii, poles, faces = _vectorized_circumradii_and_poles(pts, hull)
            curr_r = np.max(radii)
        except Exception:
            # If hull fails, we might be in a degenerate state. Break or jitter.
            break
            
        if not np.isfinite(curr_r):
            break
            
        if curr_r < best_r:
            best_r = curr_r
            best = pts.copy()
            
        # Find the worst facet
        worst_idx = np.argmax(radii)
        if not np.isfinite(radii[worst_idx]):
            break
            
        # Get the vertices and pole of the worst facet
        v_indices = faces[worst_idx]
        pole = poles[worst_idx]
        
        # Move the three vertices towards the pole
        # We use a gradient-like step. The direction to reduce the radius is towards the pole.
        # However, moving vertices directly might break the hull or create new worse facets.
        # A standard technique is to move the vertices a small fraction of the way to the pole.
        
        cand = pts.copy()
        for v in v_indices:
            # Move point v towards pole
            # Normalize the step direction
            dir_vec = pole - pts[v]
            dir_norm = np.linalg.norm(dir_vec)
            if dir_norm > _EPS:
                dir_vec /= dir_norm
            else:
                dir_vec = 0.0
            
            cand[v] = pts[v] + step * dir_vec
            # Renormalize to sphere
            cand[v] /= (np.linalg.norm(cand[v]) + _EPS)
        
        # Check if this move improved the radius
        # We don't want to recompute the full hull if it's likely to be worse, but we need to know.
        # For simplicity and robustness, we compute the new hull.
        # Optimization: We can skip the hull computation if we are confident, but it's safer to check.
        try:
            hull_cand = ConvexHull(cand)
            radii_cand, _, _ = _vectorized_circumradii_and_poles(cand, hull_cand)
            curr_r_cand = np.max(radii_cand)
        except Exception:
            curr_r_cand = np.pi
            
        if np.isfinite(curr_r_cand) and curr_r_cand < curr_r:
            pts = cand
        else:
            # If the move didn't help, try a smaller step or a random perturbation
            # Reduce step size
            step *= decay
            if step < 1e-5:
                step = 1e-5
                
            # Occasionally, do a random perturbation to escape local minima
            if rng.random() < 0.02:
                j = int(rng.integers(N))
                noise = rng.normal(size=3) * 0.02
                cand2 = pts.copy()
                cand2[j] = cand2[j] + noise
                cand2[j] /= (np.linalg.norm(cand2[j]) + _EPS)
                try:
                    hull_cand2 = ConvexHull(cand2)
                    radii_cand2, _, _ = _vectorized_circumradii_and_poles(cand2, hull_cand2)
                    curr_r_cand2 = np.max(radii_cand2)
                except Exception:
                    curr_r_cand2 = np.pi
                if np.isfinite(curr_r_cand2) and curr_r_cand2 < curr_r:
                    pts = cand2
                    step = step * 1.5 # Reward success with larger step? Or keep decay. Let's keep decay.
            
    return best, best_r

def construct():
    deadline = time.time() + 110.0  # Safety margin
    rng = np.random.default_rng(12345)
    
    # Start with Fibonacci
    best = _initial_fib()
    best_r = np.pi
    
    # Try a few different starting configurations
    starts = [best]
    
    # Random starts
    for k in range(3):
        p = rng.normal(size=(N, 3))
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        starts.append(p)
        
    # Jittered Fibonacci
    for k in range(3):
        p = _initial_fib()
        noise = rng.normal(size=(N, 3)) * 0.1
        p = p + noise
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        starts.append(p)
        
    for s in starts:
        if time.time() > deadline - 10.0:
            break
        cand, r = _improve_vectorized(s, deadline, rng)
        if np.isfinite(r) and r < best_r:
            best_r = r
            best = cand
            
    # Final normalization
    best = np.array(best, dtype=float)
    best /= (np.linalg.norm(best, axis=1, keepdims=True) + _EPS)
    
    # Validity check
    if not np.all(np.isfinite(best)) or best.shape != (N, 3):
        return _initial_fib()
        
    return best
