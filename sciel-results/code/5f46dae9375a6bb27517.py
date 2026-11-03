"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import math

def _to_sphere(pts):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _covering_radius(pts):
    """
    Compute the covering radius (in radians) using convex hull.
    The covering radius is the maximum circumradius over all facets of the convex hull.
    """
    pts = _to_sphere(pts)
    n_pts = len(pts)
    
    # Check for coincident points
    # Vectorized distance check is faster for small N, but loop is fine for N=50
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-8:
                return np.pi  # invalid - coincident points
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi  # invalid hull
    
    simplices = hull.simplices
    
    max_circumradius = 0.0
    
    for simplex in simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        # Compute circumcenter of the spherical triangle
        # The circumcenter is the normalized cross product of the edge vectors?
        # No, the circumcenter of a triangle on the unit sphere is the normalized 
        # vector sum of the vertices? No.
        # It is the point p such that p.a = p.b = p.c.
        # p is perpendicular to (b-a) and (c-a).
        # So p is parallel to (b-a) x (c-a).
        
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            # Degenerate triangle
            max_circumradius = max(max_circumradius, np.pi)
            continue
        n = n / n_norm
        
        # The circumcenter is either n or -n.
        # We want the one that is "inside" the triangle, which corresponds to the 
        # smaller angular distance to the vertices.
        
        best_radius = np.pi
        for sign in [1.0, -1.0]:
            p = sign * n
            da = np.arccos(np.clip(np.dot(p, a), -1, 1))
            db = np.arccos(np.clip(np.dot(p, b), -1, 1))
            dc = np.arccos(np.clip(np.dot(p, c), -1, 1))
            
            # The circumcenter should be equidistant. Due to floating point, check closeness.
            if abs(da - db) < 1e-5 and abs(da - dc) < 1e-5:
                if da < best_radius:
                    best_radius = da
        
        max_circumradius = max(max_circumradius, best_radius)
    
    return max_circumradius

def _fibonacci_spiral(n, offset=0.0):
    """Generate n points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n):
    """Generate n random points on the sphere."""
    pts = np.random.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _icosahedron_based_points(n):
    """
    Generate points based on icosahedron vertices and face centroids, 
    then subdivide or jitter to get n points.
    For n=50, we can start with 12 icosahedron vertices and add points
    at the centers of the 20 faces, and then subdivide the largest facets.
    This is a complex heuristic. Let's just use random and Fibonacci for simplicity
    and rely on the optimization.
    """
    return _random_points(n)

def _lloyd_relax(pts, max_iter=50, temp=0.0):
    """
    Lloyd's relaxation for sphere covering.
    Moves each point to the centroid of its Voronoi cell (approximated by 
    the average of the circumcenters of the facets it belongs to).
    """
    current_pts = pts.copy()
    current_val = _covering_radius(current_pts)
    
    for it in range(max_iter):
        try:
            hull = ConvexHull(current_pts)
        except Exception:
            # If hull fails, break or return current
            break
            
        simplices = hull.simplices
        
        # Accumulate circumcenters for each point
        # centroids[i] will be the sum of circumcenters of all facets containing point i
        centroids = np.zeros_like(current_pts)
        counts = np.zeros(len(current_pts))
        
        for simplex in simplices:
            a_idx, b_idx, c_idx = simplex
            a = current_pts[a_idx]
            b = current_pts[b_idx]
            c = current_pts[c_idx]
            
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            
            # Determine correct sign for circumcenter
            # The circumcenter should be on the same side of the triangle as the origin? 
            # No, the circumcenter is on the sphere.
            # We want the circumcenter that is "inside" the spherical triangle.
            # This is the one that minimizes the distance to the vertices.
            
            da1 = np.arccos(np.clip(np.dot(n, a), -1, 1))
            da2 = np.arccos(np.clip(np.dot(-n, a), -1, 1))
            
            # Pick the one with smaller distance to a (and b, c)
            if da1 < da2:
                cc = n
            else:
                cc = -n
            
            # Add this circumcenter to the accumulators for the vertices
            centroids[a_idx] += cc
            centroids[b_idx] += cc
            centroids[c_idx] += cc
            counts[a_idx] += 1
            counts[b_idx] += 1
            counts[c_idx] += 1
        
        # Update points
        new_pts = current_pts.copy()
        for i in range(len(current_pts)):
            if counts[i] > 0:
                # Move towards the centroid
                # If temp > 0, add some noise
                new_centroid = centroids[i] / counts[i]
                if temp > 0:
                    noise = np.random.randn(3) * temp
                    new_centroid = new_centroid + noise
                new_pts[i] = _to_sphere(new_centroid)
            else:
                # Point is not in any facet? Should not happen for convex hull.
                # Keep it or randomize?
                new_pts[i] = current_pts[i]
        
        new_val = _covering_radius(new_pts)
        
        # Accept if better, or with probability if worse (Simulated Annealing)
        if new_val < current_val or (temp > 0 and np.random.random() < np.exp(-(new_val - current_val) / temp)):
            current_pts = new_pts
            current_val = new_val
        else:
            # If we didn't accept, we might want to cool down or break if stuck
            # For now, just continue, but if temp is 0, we might be stuck.
            # If temp is 0 and we didn't improve, we can break early if we want.
            if temp == 0 and it > 10:
                # Check if we've been stuck for a while
                pass # Simple break logic omitted for now, let it run
        
    return current_pts, current_val

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_pts = None
    best_val = np.pi
    
    # 1. Fibonacci Spiral Start
    pts = _fibonacci_spiral(N)
    val = _covering_radius(pts)
    if val < best_val:
        best_val = val
        best_pts = pts.copy()
    
    # 2. Lloyd Relaxation from Fibonacci
    opt_pts, opt_val = _lloyd_relax(pts, max_iter=30, temp=0.0)
    if opt_val < best_val:
        best_val = opt_val
        best_pts = opt_pts.copy()
    
    # 3. Multiple Random Restarts with SA-Lloyd
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        pts_r = _random_points(N)
        # Initial local relaxation to get out of obvious bad spots
        opt_r, val_r = _lloyd_relax(pts_r, max_iter=10, temp=0.0)
        
        # Simulated Annealing Lloyd
        # Start with higher temp, cool down
        sa_pts, sa_val = _lloyd_relax(opt_r, max_iter=20, temp=0.1)
        
        if sa_val < best_val:
            best_val = sa_val
            best_pts = sa_pts.copy()
            
            # If we found a very good solution, try to refine it with low temp
            if time.time() < deadline - 10.0:
                ref_pts, ref_val = _lloyd_relax(sa_pts, max_iter=20, temp=0.01)
                if ref_val < best_val:
                    best_val = ref_val
                    best_pts = ref_pts.copy()
    
    # 4. Final refinement on the best solution found
    if time.time() < deadline - 5.0:
        final_pts, final_val = _lloyd_relax(best_pts, max_iter=20, temp=0.0)
        if final_val < best_val:
            best_val = final_val
            best_pts = final_pts.copy()
    
    return _to_sphere(best_pts)
# EVOLVE-BLOCK-END
