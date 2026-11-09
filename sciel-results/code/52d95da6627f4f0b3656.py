"""
Strategy: Direct Min-Max Facet Repair via Spherical Delaunay Triangulation.
Instead of using Lloyd's algorithm (which minimizes sum of squared distances and may
not minimize the max distance), we directly optimize the covering radius.

The covering radius is determined by the circumcircles of the Delaunay triangles
(facets of the convex hull). For each facet, if its circumradius is the maximum,
we move the three vertices outward (towards the circumcenter) to shrink that specific
facet, while ensuring we don't create larger facets elsewhere. We use a gradient-like
step: compute the circumcenter of the largest facet, and move its vertices towards it.
This is a local search that explicitly targets the bottleneck.

We also incorporate a "relaxation" step: after moving vertices to fix the largest facet,
we might create new large facets. We iterate this process. To avoid getting stuck in
local minima, we occasionally apply a small random perturbation (simulated annealing style).
"""

import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def _compute_circumcenters(points, facets):
    """
    Compute the circumcenters of all facets and their angular radii.
    Returns:
        centers: array of shape (F, 3)
        radii: array of shape (F,) in radians
    """
    F = len(facets)
    p1 = points[facets[:, 0]]
    p2 = points[facets[:, 1]]
    p3 = points[facets[:, 2]]
    
    v1 = p1 - p2
    v2 = p1 - p3
    c_dir = np.cross(v1, v2)
    
    norms = np.linalg.norm(c_dir, axis=1)
    centers = np.zeros_like(c_dir)
    radii = np.zeros(F)
    
    valid = norms > 1e-10
    centers[valid] = c_dir[valid] / norms[valid, np.newaxis]
    
    # Orient centers towards the triangle centroid
    centroids = (p1 + p2 + p3) / 3.0
    c_norms = np.linalg.norm(centroids, axis=1)
    valid_centroids = c_norms > 1e-10
    centroids[valid_centroids] /= c_norms[valid_centroids, np.newaxis]
    
    dots = np.sum(centers * centroids, axis=1)
    centers[dots < 0] *= -1
    
    # Compute angular radius
    cos_theta = np.sum(centers * p1, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii[valid] = np.arccos(cos_theta[valid])
    radii[~valid] = 0.0
    
    return centers, radii

def _covering_radius(points):
    """
    Compute the covering radius in degrees.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        _, radii = _compute_circumcenters(points, facets)
        return np.max(radii) * 180.0 / np.pi
    except Exception:
        return 180.0

def _repair_step(points, step_size=0.1, temperature=0.0):
    """
    Perform one repair step.
    1. Find the facet with the largest circumradius.
    2. Move its vertices towards the circumcenter.
    3. Optionally add random noise.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        centers, radii = _compute_circumcenters(points, facets)
        
        max_idx = np.argmax(radii)
        max_facet = facets[max_idx]
        center = centers[max_idx]
        
        new_points = points.copy()
        
        # Move vertices towards the center
        # We use a weighted move: the further a vertex is from the center, the more it moves?
        # Or just a fixed step?
        # Let's use a simple step towards the center.
        # To prevent oscillation, we might want to move all three vertices.
        
        for v_idx in max_facet:
            p = new_points[v_idx]
            # Move towards center
            # Use a small step size to avoid instability
            new_points[v_idx] = p + step_size * (center - p)
        
        # Add random noise if temperature > 0
        if temperature > 0:
            noise = np.random.normal(0, temperature, size=(N, 3))
            new_points += noise
        
        # Normalize
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
    
    # Simulated Annealing style search
    temperature = 0.05
    cooling_rate = 0.995
    
    for iter in range(5000):
        if time.time() > deadline:
            break
            
        # Try a repair step
        new_points = _repair_step(points, step_size=0.1, temperature=temperature)
        
        new_radius = _covering_radius(new_points)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_points = new_points.copy()
            points = new_points
        else:
            # Accept worse solutions with some probability (Metropolis criterion)
            if temperature > 0:
                delta = new_radius - best_radius
                if delta > 0:
                    accept_prob = np.exp(-delta / temperature)
                    if np.random.rand() < accept_prob:
                        points = new_points
                else:
                    points = new_points
            # If we don't accept, keep old points but maybe perturb slightly?
            # For now, just keep old points.
        
        # Cool down
        temperature *= cooling_rate
        
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Covering radius: {_covering_radius(pts)} degrees")
    print(pts.shape)
