"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

# EVOLVE-BLOCK-START

def _spherical_circumcenter(v1, v2, v3):
    """
    Compute the circumcenter and circumradius of a spherical triangle on the unit sphere.
    Returns (circumcenter_unit_vector, circumradius_in_radians).
    """
    e1 = v2 - v1
    e2 = v3 - v1
    n = np.cross(e1, e2)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        # Degenerate: return centroid
        c = (v1 + v2 + v3) / 3.0
        cn = np.linalg.norm(c)
        if cn < 1e-12:
            return np.zeros(3), 0.0
        c = c / cn
        d = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
        return c, d

    # Solve for C = v1 + u*e1 + v*e2 such that C.v1 = C.v2 = C.v3
    # (C-v1).v1 = 0 is not required; instead |C|=1 and C.v1=C.v2=C.v3
    # Equivalently: C.v2 - C.v1 = 0 and C.v3 - C.v1 = 0
    # C = v1 + u*e1 + v*e2
    # (v1 + u*e1 + v*e2).v2 - (v1 + u*e1 + v*e2).v1 = 0
    # v1.v2 + u*e1.v2 + v*e2.v2 - v1.v1 - u*e1.v1 - v*e2.v1 = 0
    # u*(e1.v2 - e1.v1) + v*(e2.v2 - e2.v1) = v1.v1 - v1.v2 = 1 - v1.v2
    # Similarly for v3.
    A = np.array([
        [e1.dot(v2) - e1.dot(v1), e2.dot(v2) - e2.dot(v1)],
        [e1.dot(v3) - e1.dot(v1), e2.dot(v3) - e2.dot(v1)]
    ])
    b = np.array([
        1.0 - v1.dot(v2),
        1.0 - v1.dot(v3)
    ])

    try:
        uv = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        c = (v1 + v2 + v3) / 3.0
        cn = np.linalg.norm(c)
        if cn < 1e-12:
            return np.zeros(3), 0.0
        c = c / cn
        d = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
        return c, d

    C = v1 + uv[0]*e1 + uv[1]*e2
    C_norm = np.linalg.norm(C)
    if C_norm < 1e-12:
        c = (v1 + v2 + v3) / 3.0
        cn = np.linalg.norm(c)
        if cn < 1e-12:
            return np.zeros(3), 0.0
        c = c / cn
        d = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
        return c, d
    C = C / C_norm

    # Circumradius: angular distance from C to v1
    dot = np.clip(np.dot(C, v1), -1.0, 1.0)
    r = np.arccos(dot)
    return C, r


def _compute_all_circumradii(points, hull):
    """Vectorized computation of circumradii for all facets."""
    simplices = hull.simplices
    n_facets = len(simplices)
    radii = np.zeros(n_facets)
    centers = np.zeros((n_facets, 3))
    
    for i in range(n_facets):
        f = simplices[i]
        C, r = _spherical_circumcenter(points[f[0]], points[f[1]], points[f[2]])
        radii[i] = r
        centers[i] = C
    
    return radii, centers


def _covering_radius(points):
    """Compute the covering radius (max circumradius) in radians."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi  # Worst case if hull fails
    radii, _ = _compute_all_circumradii(points, hull)
    return np.max(radii)


def _lloyd_step(points):
    """One step of Lloyd's algorithm on the sphere."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return points
    
    # Map each point to its adjacent facets
    point_facets = [[] for _ in range(N)]
    for f in hull.simplices:
        for idx in f:
            point_facets[idx].append(f)
    
    new_points = np.zeros_like(points)
    for i in range(N):
        if not point_facets[i]:
            new_points[i] = points[i]
            continue
        centroid = np.zeros(3)
        count = 0
        for f in point_facets[i]:
            C, r = _spherical_circumcenter(points[f[0]], points[f[1]], points[f[2]])
            centroid += C
            count += 1
        if count > 0:
            norm = np.linalg.norm(centroid)
            if norm > 1e-10:
                new_points[i] = centroid / norm
            else:
                new_points[i] = points[i]
        else:
            new_points[i] = points[i]
    return new_points


def _refine_covering(points, max_iterations=300, time_budget=60.0):
    """Iteratively reduce the covering radius by moving vertices of large facets."""
    start = time.time()
    
    for it in range(max_iterations):
        if time.time() - start > time_budget:
            break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        radii, centers = _compute_all_circumradii(points, hull)
        max_r = np.max(radii)
        
        # Find facets with radius above threshold (top few)
        threshold = max_r * 0.95
        large_facets = np.where(radii > threshold)[0]
        
        if len(large_facets) == 0:
            break
        
        # For each large facet, move its vertices toward the circumcenter
        # Use a damped step to avoid oscillation
        step_size = 0.1
        for fi in large_facets:
            f = hull.simplices[fi]
            C = centers[fi]
            for idx in f:
                v = points[idx]
                # Move toward circumcenter with damping
                new_v = v + step_size * (C - v)
                new_v = new_v / np.linalg.norm(new_v)
                points[idx] = new_v
    
    return points


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin (150s limit, 40s buffer)
    
    # 1. Initial Configuration: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Lloyd's Algorithm for uniform distribution
    lloyd_deadline = deadline - 40.0  # Leave 40s for refinement
    for _ in range(200):
        if time.time() > lloyd_deadline:
            break
        points = _lloyd_step(points)
    
    # 3. Refinement: Reduce Covering Radius
    refine_time = min(40.0, deadline - time.time())
    points = _refine_covering(points, max_iterations=500, time_budget=refine_time)
    
    # 4. Final high-precision polish: local optimization on worst facets
    polish_deadline = deadline - 5.0
    for _ in range(100):
        if time.time() > polish_deadline:
            break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        radii, centers = _compute_all_circumradii(points, hull)
        max_r = np.max(radii)
        
        # Only polish if still above ~18.5 degrees
        if max_r < np.radians(18.5):
            break
        
        # Find the single worst facet and do a finer local move
        worst_idx = np.argmax(radii)
        f = hull.simplices[worst_idx]
        C = centers[worst_idx]
        
        # Adaptive step: smaller for better convergence
        step = 0.02
        for idx in f:
            v = points[idx]
            new_v = v + step * (C - v)
            new_v = new_v / np.linalg.norm(new_v)
            points[idx] = new_v
    
    # Ensure points are normalized
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / np.maximum(norms, 1e-15)
    
    return points
# EVOLVE-BLOCK-END
