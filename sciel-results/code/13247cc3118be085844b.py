"""
Optimized sphere covering with 130 points via Spherical Lloyd's Algorithm.

Strategy:
1. Formulate the problem as minimizing the maximum circumradius of the spherical Voronoi cells.
2. Use the dual relationship between Delaunay triangulation (via ConvexHull) and Voronoi tessellation.
3. Compute the spherical centroid (Voronoi cell center) for each point using spherical barycentric coordinates.
4. Iterate: Move each point to its cell's centroid (Lloyd's step).
5. This directly targets the covering radius, unlike force-directed methods which target packing (min distance).
"""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130
_DEADLINE = time.time() + 140  # 10s safety margin

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _time_ok(margin=5.0):
    return time.time() < _DEADLINE - margin

def _covering_radius(pts):
    """Compute the covering radius in radians via convex hull facet circumcenters."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi / 2

    max_r = 0.0
    for simplex in hull.simplices:
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        v1, v2 = b - a, c - a
        cross = np.cross(v1, v2)
        cnorm = np.linalg.norm(cross)
        if cnorm < 1e-12:
            continue
        x = cross / cnorm
        centroid = (a + b + c) / 3.0
        if np.dot(x, centroid) < 0:
            x = -x
        dot_val = np.clip(np.dot(x, a), -1.0, 1.0)
        r = np.arccos(dot_val)
        if r > max_r:
            max_r = r
    return max_r

def _lloyd_step(pts):
    """
    Perform one Lloyd's algorithm step.
    1. Compute ConvexHull to get Delaunay triangulation (facets).
    2. For each point, identify its Voronoi cell vertices (the circumcenters of the Delaunay triangles sharing that point).
    3. Compute the spherical centroid of these vertices.
    4. Move the point to this centroid.
    """
    pts = _normalize(pts.copy())
    try:
        hull = ConvexHull(pts)
    except Exception:
        return pts

    # Map point index to list of facet indices containing it
    point_to_facets = [[] for _ in range(N)]
    for i, simplex in enumerate(hull.simplices):
        for idx in simplex:
            point_to_facets[idx].append(i)
            
    new_pts = np.zeros_like(pts)
    
    for i in range(N):
        facets = point_to_facets[i]
        if not facets:
            new_pts[i] = pts[i]
            continue
            
        # Collect vertices of the Voronoi cell (circumcenters of adjacent Delaunay triangles)
        # Note: We should only include facets where the circumcenter is "in front" or valid?
        # For covering radius, the relevant "holes" are the circumcenters of the Delaunay triangles.
        # The Voronoi cell of point i is the convex hull of the circumcenters of all Delaunay triangles containing i.
        # The spherical centroid of this cell is a good proxy for the "center of mass" of the empty space.
        
        vertices = []
        for f_idx in facets:
            simplex = hull.simplices[f_idx]
            a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            v1, v2 = b - a, c - a
            cross = np.cross(v1, v2)
            cnorm = np.linalg.norm(cross)
            if cnorm < 1e-12:
                continue
            x = cross / cnorm
            centroid = (a + b + c) / 3.0
            if np.dot(x, centroid) < 0:
                x = -x
            vertices.append(x)
            
        if not vertices:
            new_pts[i] = pts[i]
            continue
            
        vertices = np.array(vertices)
        
        # Spherical centroid: normalize the vector sum of the vertices
        # This is the standard approximation for the centroid of a spherical polygon (or polyhedron cell)
        center = np.sum(vertices, axis=0)
        norm = np.linalg.norm(center)
        if norm < 1e-12:
            new_pts[i] = pts[i]
        else:
            new_pts[i] = center / norm
            
    return _normalize(new_pts)

def _fibonacci_spiral(n, jitter=0.0, seed=0):
    """Fibonacci spiral on the unit sphere with optional jitter."""
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    if jitter > 0:
        i = i + rng.randn(n) * jitter
        i = np.clip(i, 0.5, n - 0.5)
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def construct():
    """
    1. Generate diverse initial configs.
    2. Run Lloyd's algorithm iterations.
    3. Return the best configuration found.
    """
    start_time = time.time()
    best_pts = None
    best_r = np.pi
    
    # Initial configs
    inits = []
    for seed in [42, 123, 7, 999, 31415, 271828, 161803, 111, 222, 333]:
        inits.append(_fibonacci_spiral(N, jitter=0.05, seed=seed))
    inits.append(_fibonacci_spiral(N, jitter=0.0, seed=0))
    rng = np.random.RandomState(555)
    inits.append(_normalize(rng.randn(N, 3)))
    
    # Sort by initial covering radius
    init_scores = []
    for p in inits:
        try:
            r = _covering_radius(p)
        except Exception:
            r = np.pi
        init_scores.append((r, p))
    init_scores.sort(key=lambda x: x[0])
    
    # Run Lloyd's algorithm on each initial config
    for r0, p0 in init_scores:
        if not _time_ok(10.0):
            break
            
        pts = p0.copy()
        
        # Iterate Lloyd's steps
        # Lloyd's can get stuck in local minima, but it's very effective for covering.
        # We run a fixed number of steps, checking time.
        for it in range(200):
            if not _time_ok(5.0):
                break
                
            pts = _lloyd_step(pts)
            
            # Optional: If we want to be more aggressive, we could add a small perturbation
            # to escape local minima, but pure Lloyd is often stable enough for N=130.
            
            # Evaluate covering radius every few steps to track best
            if it % 10 == 0:
                cr = _covering_radius(pts)
                if cr < best_r:
                    best_r = cr
                    best_pts = pts.copy()
                    
        # Final evaluation for this trajectory
        cr = _covering_radius(pts)
        if cr < best_r:
            best_r = cr
            best_pts = pts.copy()
            
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
        
    return _normalize(best_pts)

# Test
if __name__ == "__main__":
    t0 = time.time()
    pts = construct()
    print("Shape:", pts.shape)
    print("Covering radius (deg):", np.degrees(_covering_radius(pts)))
    print("Time: %.1f s" % (time.time() - t0))
