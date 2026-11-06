"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

# EVOLVE-BLOCK-START

def _spherical_circumcenter(v1, v2, v3):
    """
    Compute the circumcenter of a spherical triangle on the unit sphere.
    Returns (circumcenter, circumradius) where circumcenter is a unit vector.
    """
    e1 = v2 - v1
    e2 = v3 - v1
    n = np.cross(e1, e2)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-10:
        # Degenerate triangle, return centroid
        c = (v1 + v2 + v3) / 3.0
        cn = np.linalg.norm(c)
        if cn < 1e-10:
            return np.zeros(3), 0.0
        c = c / cn
        d = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
        return c, d
    
    n = n / n_norm
    
    # Solve for circumcenter C = v1 + u*e1 + v*e2
    # such that C.v1 = C.v2 = C.v3
    A = np.array([
        [e1.dot(v1) - e1.dot(v2), e2.dot(v1) - e2.dot(v2)],
        [e1.dot(v1) - e1.dot(v3), e2.dot(v1) - e2.dot(v3)]
    ])
    b = np.array([
        v1.dot(v2) - 1.0,
        v1.dot(v3) - 1.0
    ])
    
    try:
        uv = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        c = (v1 + v2 + v3) / 3.0
        cn = np.linalg.norm(c)
        if cn < 1e-10:
            return np.zeros(3), 0.0
        c = c / cn
        d = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
        return c, d
        
    C = v1 + uv[0]*e1 + uv[1]*e2
    C_norm = np.linalg.norm(C)
    if C_norm < 1e-10:
        return np.zeros(3), 0.0
    C = C / C_norm
    
    # Compute angular radius
    dot = np.clip(np.dot(C, v1), -1.0, 1.0)
    r = np.arccos(dot)
    
    return C, r


def _covering_radius(points):
    """
    Compute the covering radius in radians.
    The covering radius is the maximum over all facets of the circumradius of the facet.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi  # Invalid configuration
        
    max_r = 0.0
    for f in hull.simplices:
        _, r = _spherical_circumcenter(points[f[0]], points[f[1]], points[f[2]])
        if r > max_r:
            max_r = r
            
    return max_r


def _lloyd_step(points):
    """
    One step of Lloyd's algorithm on the sphere.
    Moves each point to the centroid of its Voronoi cell.
    The Voronoi cell centroid is approximated by the average of the circumcenters
    of the adjacent facets.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return points
        
    # Map each point to its adjacent facets
    point_facets = {i: [] for i in range(N)}
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
            C, _ = _spherical_circumcenter(points[f[0]], points[f[1]], points[f[2]])
            if np.linalg.norm(C) > 1e-10:
                centroid += C
                count += 1
                
        if count > 0:
            centroid /= count
            norm = np.linalg.norm(centroid)
            if norm > 1e-10:
                new_points[i] = centroid / norm
            else:
                new_points[i] = points[i]
        else:
            new_points[i] = points[i]
            
    return new_points


def _refine_covering(points, max_iterations=300):
    """
    Refine the configuration to minimize the covering radius.
    Iteratively finds the facet with the largest circumradius and moves its
    vertices towards the circumcenter.
    """
    best_points = points.copy()
    best_r = _covering_radius(points)
    
    for it in range(max_iterations):
        try:
            hull = ConvexHull(points)
        except Exception:
            break
            
        # Find the facet with the largest circumradius
        max_r = 0.0
        max_facet = None
        max_cc = None
        
        for idx, f in enumerate(hull.simplices):
            C, r = _spherical_circumcenter(points[f[0]], points[f[1]], points[f[2]])
            if r > max_r:
                max_r = r
                max_facet = f
                max_cc = C
                
        if max_facet is None or max_cc is None:
            break
            
        # Adaptive step size: smaller for larger radii to avoid oscillation
        # The maximum possible step to reach the circumcenter is 1.0
        # We use a fraction of the distance
        step = 0.1 + 0.1 * np.cos(it * 0.05)  # Oscillating step size
        
        # Move vertices of the worst facet towards the circumcenter
        new_points = points.copy()
        for idx in max_facet:
            v = new_points[idx]
            # Move towards C
            new_v = v + step * (max_cc - v)
            new_v = new_v / np.linalg.norm(new_v)
            new_points[idx] = new_v
            
        # Check if this improves the covering radius
        new_r = _covering_radius(new_points)
        if new_r < best_r - 1e-10:
            best_r = new_r
            best_points = new_points.copy()
            points = new_points
        else:
            # If it doesn't improve, try a smaller step or break
            # To avoid getting stuck, we can try a random perturbation
            # or just break if we've made no progress for a while
            if it > 50:
                break
                
    return best_points


def _targeted_facet_refine(points, max_iterations=500):
    """
    Targeted local search: find the worst facet, then perform a micro-polish
    on its three vertices to minimize that facet's circumradius.
    Uses gradient-free line search along geodesic bisectors.
    """
    best_points = points.copy()
    best_r = _covering_radius(points)
    no_improve_count = 0
    
    for it in range(max_iterations):
        # Find the worst facet
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        max_r = 0.0
        max_facet = None
        max_cc = None
        
        for f in hull.simplices:
            C, r = _spherical_circumcenter(points[f[0]], points[f[1]], points[f[2]])
            if r > max_r:
                max_r = r
                max_facet = f
                max_cc = C
        
        if max_facet is None or max_cc is None:
            break
        
        # Micro-polish: try small perturbations of the three vertices
        # Move each vertex slightly towards the circumcenter
        improved = False
        for trial in range(3):
            new_points = points.copy()
            for idx in max_facet:
                v = new_points[idx]
                # Geodesic interpolation towards circumcenter
                theta = 0.05 / (trial + 1)  # Decreasing step
                # Slerp: new = sin((1-t)*d)/sin(d)*v + sin(t*d)/sin(d)*C
                d = np.arccos(np.clip(np.dot(v, max_cc), -1.0, 1.0))
                if d < 1e-10:
                    continue
                t = theta / d
                t = np.clip(t, 0.0, 1.0)
                sin_d = np.sin(d)
                new_v = (np.sin((1-t)*d) / sin_d) * v + (np.sin(t*d) / sin_d) * max_cc
                new_v = new_v / np.linalg.norm(new_v)
                new_points[idx] = new_v
            
            new_r = _covering_radius(new_points)
            if new_r < best_r - 1e-12:
                best_r = new_r
                best_points = new_points.copy()
                points = new_points
                improved = True
                no_improve_count = 0
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 20:
                break
    
    return best_points


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Initial Configuration: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Lloyd's Algorithm (Centroidal Voronoi on Sphere)
    # Run for a fixed number of iterations or until time allows
    for _ in range(50):
        if time.time() > deadline - 60:
            break
        points = _lloyd_step(points)
        
    # 3. Refinement: Reduce Covering Radius
    if time.time() < deadline - 10:
        points = _refine_covering(points, max_iterations=200)
    
    # 4. Targeted facet refinement with micro-polish
    if time.time() < deadline - 5:
        points = _targeted_facet_refine(points, max_iterations=500)
        
    # Final check and normalization
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    return points

# EVOLVE-BLOCK-END
