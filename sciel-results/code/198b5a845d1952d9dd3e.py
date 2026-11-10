"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _covering_radius(pts):
    """Compute covering radius as max angular distance to nearest point."""
    # Use convex hull to find facets; covering radius = max circumradius of facets
    try:
        hull = ConvexHull(pts)
        max_r = 0.0
        for simplex in hull.simplices:
            # Get 3 points forming a facet
            p0, p1, p2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            # Compute circumcenter of triangle on sphere
            # The circumcenter is the point equidistant from all 3 vertices
            # Solve for center c such that |c - p0| = |c - p1| = |c - p2|
            # and c is on the sphere (|c| = 1)
            # Use the formula: circumcenter = (p0 + p1 + p2) / |p0 + p1 + p2| (for equidistant case)
            # More precisely, solve the linear system
            # c · (p0 - p1) = 0, c · (p0 - p2) = 0, |c| = 1
            v1 = p0 - p1
            v2 = p0 - p2
            # c is perpendicular to v1 and v2, so c is parallel to v1 × v2
            cross = np.cross(v1, v2)
            if np.linalg.norm(cross) < 1e-10:
                continue
            c = cross / np.linalg.norm(cross)
            # Check if c is on the correct side (outward normal)
            # The circumradius is the angular distance from c to p0
            dot = np.clip(np.dot(c, p0), -1.0, 1.0)
            r = np.arccos(dot)
            max_r = max(max_r, r)
        return np.degrees(max_r)
    except Exception:
        return 180.0

def _local_optimize(pts, max_time=60.0):
    """Local optimization to reduce covering radius."""
    start_time = time.time()
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    lr = 0.01
    n_iter = 0
    while time.time() - start_time < max_time:
        # Compute gradient by finite differences
        grad = np.zeros_like(pts)
        for i in range(N):
            for d in range(3):
                pts_pert = pts.copy()
                pts_pert[i, d] += 1e-5
                pts_pert = _normalize(pts_pert)
                cr_pert = _covering_radius(pts_pert)
                grad[i, d] = (cr_pert - best_cr) / 1e-5
        
        # Project gradient to tangent space
        for i in range(N):
            grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
        
        # Update
        pts_new = pts - lr * grad
        pts_new = _normalize(pts_new)
        
        cr_new = _covering_radius(pts_new)
        if cr_new < best_cr:
            best_cr = cr_new
            best_pts = pts_new.copy()
            pts = pts_new
            lr = min(lr * 1.1, 0.1)
        else:
            lr *= 0.8
            if lr < 1e-6:
                break
        n_iter += 1
        if n_iter % 100 == 0:
            print(f"Iter {n_iter}: CR = {best_cr:.6f} deg, lr = {lr:.6f}")
    
    return best_pts, best_cr

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Local optimization
    pts, cr = _local_optimize(pts, max_time=120.0)
    
    # Check deadline
    if time.time() > deadline:
        return _normalize(pts)
    
    return _normalize(pts)
# EVOLVE-BLOCK-END
