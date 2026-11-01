"""Improved covering: Lloyd + facet circumradius refinement + gradient-free polish."""
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay
from itertools import combinations

N = 130
TIME_LIMIT = 115.0  # safety margin before 120s kill

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _covering_radius(pts):
    """Compute covering radius in degrees via convex hull facets."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi  # fallback: worst case
    
    # For each facet, find the circumradius (angular)
    # The covering radius is the max over facets of the circumradius
    # A facet's circumradius is the angular distance from the facet plane to the sphere
    # i.e., arccos(n · c) where c is the centroid direction... 
    # Actually, for a spherical cap covering, the worst point is the pole of the facet
    # The angular distance from the pole to the farthest vertex of the facet
    
    # Better approach: for each facet, compute the circumcenter on the sphere
    # and find the max angular distance to facet vertices
    # The circumcenter is along the normal from the origin through the facet centroid
    # but we need the actual circumcenter of the spherical polygon
    
    # Simpler: the covering radius for a facet is determined by its vertices
    # The worst point in the Voronoi cell corresponding to a facet is the circumcenter
    # of that facet (if it lies within the facet's spherical polygon)
    
    # For a triangle facet with vertices a, b, c on unit sphere:
    # The circumcenter (on sphere) is in the direction of (a × b) × c normalized... 
    # Actually, the circumcenter of a spherical triangle is found by intersection of perpendicular bisectors
    # For a planar triangle, circumcenter = intersection of perpendicular bisectors
    # For spherical, we can use: the pole direction is the normalized cross product approach
    
    # Let's use a different approach: for each facet, sample the circumcenter
    # The covering radius contribution from a facet is the max angular distance
    # from the facet's circumcenter to its vertices
    
    # For a triangle with vertices v1, v2, v3 on unit sphere:
    # The circumcenter direction is proportional to:
    # (v1 * |v2-v3|^2) + (v2 * |v3-v1|^2) + (v3 * |v1-v2|^2) ... no, that's centroid
    
    # Correct spherical circumcenter: solve for point x on sphere equidistant from all 3
    # x · v1 = x · v2 = x · v3 = cos(r)
    # This means x is in the null space of [v1-v2, v1-v3] in some sense
    # x is perpendicular to (v1-v2) and (v1-v3) in the tangent plane... 
    # Actually: x · v1 = x · v2 => x · (v1-v2) = 0
    # x · v1 = x · v3 => x · (v1-v3) = 0
    # So x is in the direction of (v1-v2) × (v1-v3)
    
    max_rad = 0.0
    facets = hull.simplices  # for 3D convex hull, these are triangles
    
    for tri in facets:
        v1, v2, v3 = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        
        # Circumcenter direction
        d1 = v1 - v2
        d2 = v1 - v3
        cross = np.cross(d1, d2)
        cnorm = np.linalg.norm(cross)
        if cnorm < 1e-15:
            continue
        cc_dir = cross / cnorm
        
        # Make sure it points outward (same side as facet normal)
        # The facet normal from convex hull
        n1 = np.cross(v2 - v1, v3 - v1)
        nnorm = np.linalg.norm(n1)
        if nnorm > 1e-15:
            n1 = n1 / nnorm
            if np.dot(cc_dir, n1) < 0:
                cc_dir = -cc_dir
        
        # Angular distance from circumcenter to each vertex
        cos_d1 = np.clip(np.dot(cc_dir, v1), -1, 1)
        cos_d2 = np.clip(np.dot(cc_dir, v2), -1, 1)
        cos_d3 = np.clip(np.dot(cc_dir, v3), -1, 1)
        r1 = np.arccos(cos_d1)
        r2 = np.arccos(cos_d2)
        r3 = np.arccos(cos_d3)
        rad = max(r1, r2, r3)
        max_rad = max(max_rad, rad)
    
    return max_rad

def _covering_radius_fast(pts):
    """Faster covering radius using Delaunay triangulation on sphere."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi
    
    facets = hull.simplices
    max_rad = 0.0
    
    for tri in facets:
        v1, v2, v3 = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        
        # Spherical circumcenter
        d1 = v1 - v2
        d2 = v1 - v3
        cross = np.cross(d1, d2)
        cnorm = np.linalg.norm(cross)
        if cnorm < 1e-15:
            continue
        cc_dir = cross / cnorm
        
        # Check outward
        n1 = np.cross(v2 - v1, v3 - v1)
        nnorm = np.linalg.norm(n1)
        if nnorm > 1e-15:
            n1 = n1 / nnorm
            if np.dot(cc_dir, n1) < 0:
                cc_dir = -cc_dir
        
        # Max angular distance to vertices
        cos_vals = np.array([np.dot(cc_dir, v1), np.dot(cc_dir, v2), np.dot(cc_dir, v3)])
        cos_vals = np.clip(cos_vals, -1, 1)
        rads = np.arccos(cos_vals)
        rad = np.max(rads)
        if rad > max_rad:
            max_rad = rad
    
    return max_rad

def _lloyd_iteration(pts, n_iter=20):
    """Lloyd's algorithm for spherical Voronoi centroids."""
    pts = _normalize(pts)
    
    for _ in range(n_iter):
        # Compute Voronoi cells via Delaunay
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # For each point, find its Voronoi cell (intersection of half-spaces)
        # Approximate: use the dual of Delaunay
        # In 3D, the Voronoi cell of a point on the sphere can be approximated
        # by the spherical polygon formed by circumcenters of adjacent facets
        
        new_pts = np.zeros_like(pts)
        
        # Find facets for each point
        point_to_facets = {i: [] for i in range(N)}
        for idx, tri in enumerate(hull.simplices):
            for v in tri:
                point_to_facets[v].append(idx)
        
        for i in range(N):
            facets_i = point_to_facets[i]
            if len(facets_i) < 3:
                new_pts[i] = pts[i]
                continue
            
            # Collect circumcenters of adjacent facets
            cc_list = []
            for fidx in facets_i:
                tri = hull.simplices[fidx]
                v1, v2, v3 = pts[tri[0]], pts[tri[1]], pts[tri[2]]
                
                d1 = v1 - v2
                d2 = v1 - v3
                cross = np.cross(d1, d2)
                cnorm = np.linalg.norm(cross)
                if cnorm < 1e-15:
                    continue
                cc_dir = cross / cnorm
                
                # Outward check
                n1 = np.cross(v2 - v1, v3 - v1)
                nnorm = np.linalg.norm(n1)
                if nnorm > 1e-15:
                    n1 = n1 / nnorm
                    if np.dot(cc_dir, n1) < 0:
                        cc_dir = -cc_dir
                
                cc_list.append(cc_dir)
            
            if len(cc_list) >= 3:
                # Spherical centroid: average and normalize
                # Weighted by solid angle or just simple average
                avg = np.mean(cc_list, axis=0)
                anorm = np.linalg.norm(avg)
                if anorm > 1e-15:
                    new_pts[i] = avg / anorm
                else:
                    new_pts[i] = pts[i]
            else:
                new_pts[i] = pts[i]
        
        new_pts = _normalize(new_pts)
        # Check for degeneracy
        if np.any(np.linalg.norm(new_pts, axis=1) < 1e-10):
            break
        pts = new_pts
    
    return pts

def _gradient_free_refine(pts, n_steps=500, step_size=0.05):
    """Nelder-Mead-like or random perturbation refinement targeting covering radius."""
    pts = _normalize(pts)
    best_pts = pts.copy()
    best_rad = _covering_radius_fast(pts)
    
    for step in range(n_steps):
        # Find the worst facet
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        facets = hull.simplices
        worst_rad = 0
        worst_tri = None
        
        for tri in facets:
            v1, v2, v3 = pts[tri[0]], pts[tri[1]], pts[tri[2]]
            d1 = v1 - v2
            d2 = v1 - v3
            cross = np.cross(d1, d2)
            cnorm = np.linalg.norm(cross)
            if cnorm < 1e-15:
                continue
            cc_dir = cross / cnorm
            n1 = np.cross(v2 - v1, v3 - v1)
            nnorm = np.linalg.norm(n1)
            if nnorm > 1e-15:
                n1 = n1 / nnorm
                if np.dot(cc_dir, n1) < 0:
                    cc_dir = -cc_dir
            cos_vals = np.array([np.dot(cc_dir, v1), np.dot(cc_dir, v2), np.dot(cc_dir, v3)])
            cos_vals = np.clip(cos_vals, -1, 1)
            rads = np.arccos(cos_vals)
            rad = np.max(rads)
            if rad > worst_rad:
                worst_rad = rad
                worst_tri = tri
        
        if worst_tri is None:
            break
        
        # Perturb the vertices of the worst facet inward (toward circumcenter)
        v1, v2, v3 = pts[worst_tri[0]], pts[worst_tri[1]], pts[worst_tri[2]]
        d1 = v1 - v2
        d2 = v1 - v3
        cross = np.cross(d1, d2)
        cnorm = np.linalg.norm(cross)
        if cnorm < 1e-15:
            break
        cc_dir = cross / cnorm
        n1 = np.cross(v2 - v1, v3 - v1)
        nnorm = np.linalg.norm(n1)
        if nnorm > 1e-15:
            n1 = n1 / nnorm
            if np.dot(cc_dir, n1) < 0:
                cc_dir = -cc_dir
        
        # Move each vertex slightly toward the circumcenter
        for vi in worst_tri:
            pts[vi] = pts[vi] + step_size * (cc_dir - pts[vi])
        
        pts = _normalize(pts)
        new_rad = _covering_radius_fast(pts)
        
        if new_rad < best_rad:
            best_rad = new_rad
            best_pts = pts.copy()
        else:
            # Revert
            pts = best_pts.copy()
            # Reduce step size
            step_size *= 0.95
    
    return best_pts

def construct():
    start_time = time.time()
    
    # Initial: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_rad = _covering_radius_fast(pts)
    
    # Stage 1: Lloyd iterations
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 30:
        pts = _lloyd_iteration(pts, n_iter=30)
        rad = _covering_radius_fast(pts)
        if rad < best_rad:
            best_rad = rad
            best_pts = pts.copy()
    
    # Stage 2: More Lloyd
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 20:
        pts = _lloyd_iteration(pts, n_iter=20)
        rad = _covering_radius_fast(pts)
        if rad < best_rad:
            best_rad = rad
            best_pts = pts.copy()
    
    # Stage 3: Gradient-free refinement targeting worst facets
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 15:
        pts = _gradient_free_refine(best_pts, n_steps=300, step_size=0.03)
        rad = _covering_radius_fast(pts)
        if rad < best_rad:
            best_rad = rad
            best_pts = pts.copy()
    
    # Final: one more Lloyd pass
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 10:
        pts = _lloyd_iteration(best_pts, n_iter=10)
        rad = _covering_radius_fast(pts)
        if rad < best_rad:
            best_rad = rad
            best_pts = pts.copy()
    
    return best_pts
