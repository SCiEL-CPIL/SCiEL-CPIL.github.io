"""Optimized sphere covering with 130 points."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree
import math

N = 130
DEADLINE = time.time() + 140  # 10s safety margin

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _covering_radius(pts):
    """Compute the covering radius in radians."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi / 2  # fallback
    
    # For each facet, compute the spherical circumcenter
    max_r = 0.0
    for simplex in hull.simplices:
        p = pts[simplex]
        # Spherical circumcenter: find point equidistant from all vertices
        # For a triangle on the unit sphere, the circumcenter is the normalized
        # cross product of (p1-p0) and (p2-p0) adjusted... 
        # Actually, for 3 points on unit sphere, the circumcenter is:
        # normalize( a*(|b|^2 - |c|^2) + b*(|c|^2 - |a|^2) + c*(|a|^2 - |b|^2) )
        # But since all on unit sphere, |a|=|b|=|c|=1, so:
        # circumcenter = normalize( a*(|b|^2 - |c|^2) + ... ) doesn't work directly
        # 
        # Better: The circumcenter of 3 points on a sphere is the intersection of the sphere
        # with the plane through the 3 points... no, that's not right either.
        #
        # For 3 points a, b, c on the unit sphere, the spherical circumcenter is:
        # The point x on the sphere such that |x-a| = |x-b| = |x-c|
        # This means x·a = x·b = x·c
        # So x is in the null space of [a-b; b-c] (in the sense that x·(a-b)=0 and x·(b-c)=0)
        # So x is parallel to (a-b) × (b-c)
        
        a, b, c = p[0], p[1], p[2]
        # x is parallel to (b-a) × (c-a)
        v1 = b - a
        v2 = c - a
        cross = np.cross(v1, v2)
        cnorm = np.linalg.norm(cross)
        if cnorm < 1e-12:
            continue
        x = cross / cnorm
        
        # Check which side: we want the circumcenter that is "inside" the triangle
        # The angular distance from x to a
        dot_val = np.dot(x, a)
        dot_val = np.clip(dot_val, -1.0, 1.0)
        r = np.arccos(dot_val)
        
        # Also check the antipodal point
        x2 = -x
        dot_val2 = np.dot(x2, a)
        dot_val2 = np.clip(dot_val2, -1.0, 1.0)
        r2 = np.arccos(dot_val2)
        
        # The covering radius for this facet is the smaller one (the one on the same side)
        # Actually, we want the farthest point in the Voronoi cell, which is the circumcenter
        # that is on the "outside" of the triangle (away from the origin side)
        # Let's use the one with larger r if both are valid, but typically the circumcenter
        # of the spherical triangle is the one that maximizes distance
        r_facet = max(r, r2) if r + r2 < np.pi else min(r, r2)
        # Hmm, let me think again. The Voronoi vertex is the circumcenter. 
        # For a triangle on the sphere, there are two circumcenters (antipodal). 
        # The relevant one is the one that is "outside" the convex hull facet.
        # Since the facet is on the convex hull, the outward normal direction is what matters.
        
        # The outward normal of the facet is the normalized cross product
        # The circumcenter that is in the direction of the outward normal is the one we want
        normal = cross / cnorm
        # Check if normal points outward: dot with centroid of facet
        centroid = (a + b + c) / 3.0
        if np.dot(normal, centroid) < 0:
            normal = -normal
        
        # The circumcenter in the direction of the outward normal
        if np.dot(x, normal) < 0:
            x = -x
        
        dot_val = np.dot(x, a)
        dot_val = np.clip(dot_val, -1.0, 1.0)
        r = np.arccos(dot_val)
        max_r = max(max_r, r)
    
    return max_r

def _covering_radius_fast(pts):
    """Faster covering radius using kDTree to find nearest neighbors and check circumcenters."""
    pts = _normalize(pts)
    # Use the fact that covering radius = max over all points y of min_i angular_dist(y, p_i)
    # This is equivalent to the max circumradius of Delaunay triangles
    
    # For efficiency, use the convex hull approach but with vectorized operations
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi / 2
    
    max_r = 0.0
    simplices = hull.simplices
    for simplex in simplices:
        p = pts[simplex]
        a, b, c = p[0], p[1], p[2]
        v1 = b - a
        v2 = c - a
        cross = np.cross(v1, v2)
        cnorm = np.linalg.norm(cross)
        if cnorm < 1e-12:
            continue
        x = cross / cnorm
        
        # Determine correct orientation
        centroid = (a + b + c) / 3.0
        if np.dot(x, centroid) < 0:
            x = -x
        
        dot_val = np.dot(x, a)
        dot_val = np.clip(dot_val, -1.0, 1.0)
        r = np.arccos(dot_val)
        if r > max_r:
            max_r = r
    
    return max_r

def _initial_config():
    """Generate initial configuration using Fibonacci spiral with some randomization."""
    # Use multiple random seeds and pick the best
    best_pts = None
    best_r = np.pi
    rng = np.random.RandomState(42)
    
    for trial in range(5):
        # Fibonacci spiral with slight perturbation
        i = np.arange(N) + 0.5 + rng.randn(N) * 0.1
        i = np.clip(i, 0.5, N - 0.5)
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i + rng.randn(N) * 0.05
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        pts = _normalize(pts)
        
        cr = _covering_radius_fast(pts)
        if cr < best_r:
            best_r = cr
            best_pts = pts.copy()
    
    return best_pts

def _local_optimize(pts, max_time=30):
    """Local optimization using gradient descent on covering radius."""
    start_time = time.time()
    pts = _normalize(pts.copy())
    best_pts = pts.copy()
    best_r = _covering_radius_fast(pts)
    
    # Use a subgradient method: for each point, compute the direction to move
    # based on the farthest point in its Voronoi cell
    
    # First, compute the Voronoi diagram using the convex hull
    # For each point, find which facets it's part of, then the circumcenters
    
    # Simpler approach: use repulsion + attraction to circumcenters
    # For each point i, the "pressure" is from the circumcenters of adjacent triangles
    # We want to move point i toward the circumcenter that is farthest
    
    # Alternative: use the fact that at optimum, each point should be equidistant
    # from the farthest points in its Voronoi cell
    
    # Let's use a simple iterative method:
    # 1. Compute covering radius
    # 2. Find the "bottleneck" - the point y that is farthest from all points
    # 3. Move the nearest point(s) toward y
    
    step_size = 0.01
    for iteration in range(2000):
        if time.time() - start_time > max_time:
            break
        if time.time() > DEADLINE - 5:
            break
        
        pts = _normalize(pts)
        cr = _covering_radius_fast(pts)
        if cr < best_r:
            best_r = cr
            best_pts = pts.copy()
        
        # Find the worst point: sample the sphere and find the point with max min-distance
        # For efficiency, use the circumcenters of the Delaunay triangles
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # For each facet, compute circumcenter and distance to nearest point
        worst_dist = 0
        worst_y = None
        worst_nearest_idx = None
        
        for simplex in hull.simplices:
            p = pts[simplex]
            a, b, c = p[0], p[1], p[2]
            v1 = b - a
            v2 = c - a
            cross = np.cross(v1, v2)
            cnorm = np.linalg.norm(cross)
            if cnorm < 1e-12:
                continue
            x = cross / cnorm
            centroid = (a + b + c) / 3.0
            if np.dot(x, centroid) < 0:
                x = -x
            
            # Distance from x to all points
            dots = np.clip(pts @ x, -1.0, 1.0)
            dists = np.arccos(dots)
            min_dist = np.min(dists)
            min_idx = np.argmin(dists)
            
            if min_dist > worst_dist:
                worst_dist = min_dist
                worst_y = x.copy()
                worst_nearest_idx = min_idx
        
        if worst_y is None:
            break
        
        # Move the nearest point toward the worst point
        # Also move nearby points slightly
        move_dir = worst_y - pts[worst_nearest_idx]
        move_dir = _normalize(move_dir.reshape(1, -1)).flatten()
        
        # Adaptive step size
        if cr > 0.2:  # ~11.5 degrees
            step = 0.005
        elif cr > 0.15:
            step = 0.002
        else:
            step = 0.0005
        
        pts[worst_nearest_idx] += step * move_dir
        
        # Also add small random perturbation to escape local minima
        if iteration % 100 == 50:
            rng = np.random.RandomState(iteration)
            perturbation = rng.randn(N, 3) * 0.001
            pts += perturbation
    
    return best_pts, best_r

def _anneal_refine(pts, max_time=20):
    """Simulated annealing style refinement."""
    start_time = time.time()
    pts = _normalize(pts.copy())
    best_pts = pts.copy()
    best_r = _covering_radius_fast(pts)
    
    T = 0.01  # temperature
    cooling = 0.99
    
    while time.time() - start_time < max_time and time.time() < DEADLINE - 5:
        T *= cooling
        if T < 0.0001:
            break
        
        # Random move: pick a random point and move it slightly
        rng = np.random.RandomState(int(time.time() * 1000) % 2**31)
        idx = rng.randint(N)
        direction = rng.randn(3)
        direction = direction / np.linalg.norm(direction)
        
        new_pts = pts.copy()
        new_pts[idx] += T * 0.1 * direction
        new_pts = _normalize(new_pts)
        
        new_r = _covering_radius_fast(new_pts)
        
        # Accept if better or with probability exp(-delta/T)
        if new_r < best_r:
            best_r = new_r
            best_pts = new_pts.copy()
        
        delta = new_r - _covering_radius_fast(pts)
        if delta < 0 or rng.random() < np.exp(-delta / T):
            pts = new_pts
    
    return best_pts, best_r

def construct():
    """Main function to construct the optimal configuration."""
    start_time = time.time()
    
    # Step 1: Get initial configuration
    pts = _initial_config()
    
    # Step 2: Local optimization
    if time.time() < DEADLINE - 40:
        pts, cr = _local_optimize(pts, max_time=30)
    
    # Step 3: Annealing refinement
    if time.time() < DEADLINE - 15:
        pts, cr = _anneal_refine(pts, max_time=20)
    
    # Final normalization
    pts = _normalize(pts)
    
    return pts

# Test
if __name__ == "__main__":
    pts = construct()
    print("Shape:", pts.shape)
    print("Covering radius (deg):", np.degrees(_covering_radius_fast(pts)))
