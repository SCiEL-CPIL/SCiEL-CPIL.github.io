"""
Strategy: Direct Min-Max Facet Repair via Spherical Delaunay Triangulation.
Instead of using Lloyd's algorithm (which minimizes sum of squared distances and may
not minimize the max distance), we directly optimize the covering radius.

The covering radius is determined by the circumcircles of the Delaunay triangles
(facets of the convex hull). For each facet, if its circumradius is the maximum,
we move the three vertices of that facet along the geodesics towards the
circumcenter. This is a direct gradient step for the min-max objective.

We use a steepest-descent-like approach:
1. Compute ConvexHull.
2. For each facet, compute the circumcenter and circumradius.
3. Identify the "worst" facet (largest circumradius).
4. Move the 3 vertices of the worst facet towards its circumcenter by a small step.
5. To speed up convergence, we can move all facets simultaneously with steps
   proportional to how much their circumradius exceeds the median or a threshold.
   Or, we can use a batch update: for each vertex, compute a weighted average of
   the directions to the circumcenters of incident facets, weighted by the
   circumradius of that facet.

Let's use the batch update:
For each vertex v, let F(v) be the set of incident facets.
For each facet f in F(v), let c_f be the circumcenter and r_f be the circumradius.
The direction to move v is towards c_f.
We can compute a "force" on v:
F_v = sum_{f in F(v)} w_f * (c_f - v) / ||c_f - v||  (normalized direction)
where w_f could be r_f or (r_f - r_avg)^2 or similar.
Then update v <- v + step * F_v, then normalize to sphere.

This is analogous to repulsion but towards the "holes" (circumcenters).
"""

import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def _compute_circumcenters(points, facets):
    """
    Compute circumcenters and circumradii for all facets.
    Returns:
        centers: array of shape (n_facets, 3)
        radii: array of shape (n_facets,)
    """
    n_facets = facets.shape[0]
    centers = np.zeros((n_facets, 3))
    radii = np.zeros(n_facets)
    
    p1 = points[facets[:, 0]]
    p2 = points[facets[:, 1]]
    p3 = points[facets[:, 2]]
    
    # Vectorized computation
    v1 = p1 - p2
    v2 = p1 - p3
    c_dir = np.cross(v1, v2)
    
    norms = np.linalg.norm(c_dir, axis=1)
    valid = norms > 1e-10
    
    centers[valid] = c_dir[valid] / norms[valid, np.newaxis]
    
    # Orient centers to be on the same side as the triangle centroid
    centroids = (p1 + p2 + p3) / 3.0
    c_norms = np.linalg.norm(centroids, axis=1)
    valid_centroids = c_norms > 1e-10
    centroids[valid_centroids] /= c_norms[valid_centroids, np.newaxis]
    
    dots = np.sum(centers * centroids, axis=1)
    centers[dots < 0] *= -1
    
    # Compute circumradius: angular distance from center to p1
    cos_theta = np.sum(centers[valid] * p1[valid], axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii[valid] = np.arccos(cos_theta)
    
    return centers, radii

def _covering_radius(points):
    """
    Compute the covering radius in degrees.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        centers, radii = _compute_circumcenters(points, facets)
        return np.max(radii) * 180.0 / np.pi
    except Exception:
        return 180.0

def _repair_step(points, step_size=0.05):
    """
    Perform one repair step.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        centers, radii = _compute_circumcenters(points, facets)
        
        n_points = points.shape[0]
        forces = np.zeros_like(points)
        
        # For each facet, add force to its 3 vertices
        # Weight by the circumradius squared to emphasize large holes
        weights = radii ** 2
        
        for i in range(facets.shape[0]):
            f = facets[i]
            c = centers[i]
            w = weights[i]
            
            if w < 1e-10:
                continue
                
            for v_idx in f:
                # Direction from point to circumcenter
                direction = c - points[v_idx]
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    forces[v_idx] += w * direction / norm
        
        # Normalize forces to prevent large jumps
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        valid_forces = force_norms > 1e-10
        forces[valid_forces] /= force_norms[valid_forces]
        
        new_points = points + step_size * forces
        
        # Normalize to sphere
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        new_points = new_points / norms
        
        return new_points
        
    except Exception:
        return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Adaptive step size
    step_size = 0.1
    
    for iter in range(10000):
        if time.time() > deadline:
            break
            
        # Perform repair step
        new_points = _repair_step(points, step_size)
        
        # Check new radius
        new_radius = _covering_radius(new_points)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_points = new_points.copy()
            points = new_points
            # Reduce step size slightly on success to fine-tune
            step_size = max(0.01, step_size * 0.99)
        else:
            # If radius increased, revert and reduce step size
            step_size = max(0.01, step_size * 0.9)
            
        # Optional: Add small random perturbation if stuck
        # We can detect stagnation by checking if best_radius hasn't improved in last K steps
        # For now, let's just run.
        
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Covering radius: {_covering_radius(pts)} degrees")
    print(pts.shape)
