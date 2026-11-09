"""
Strategy: Global Stochastic Search with Simulated Annealing.
The current program uses Lloyd's algorithm, which minimizes the sum of squared
distances (k-means objective) and gets trapped in local minima where the
covering radius (max distance) is not optimal. 

This program uses Simulated Annealing (SA) to directly minimize the covering
radius. 
1. We initialize with a Fibonacci spiral.
2. We iteratively pick a random point and perturb it by a small random vector on the sphere.
3. We calculate the change in the covering radius.
4. We accept the move if it improves the radius, or with a probability 
   exp(-delta / T) if it worsens it, where T is the current temperature.
5. We cool down T over time.
This allows escaping local minima and directly optimizes the target metric.
"""

import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def _covering_radius(points):
    """
    Compute the covering radius in degrees.
    Vectorized computation of circumradii of convex hull facets.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        p1 = points[facets[:, 0]]
        p2 = points[facets[:, 1]]
        p3 = points[facets[:, 2]]
        
        # Normal to the plane defined by p1, p2, p3
        # C is the intersection of the sphere and the line perpendicular to the 
        # plane of the triangle passing through the origin? No.
        
        # The circumcenter of a spherical triangle is the unit vector c such that
        # c.p1 = c.p2 = c.p3.
        # This implies c is orthogonal to (p1-p2) and (p1-p3).
        # So c is parallel to cross(p1-p2, p1-p3).
        
        v1 = p1 - p2
        v2 = p1 - p3
        c_dir = np.cross(v1, v2)
        
        norms = np.linalg.norm(c_dir, axis=1)
        
        # Handle degenerate cases
        valid = norms > 1e-10
        if not np.any(valid):
            return 90.0 # Fallback
            
        c = np.zeros_like(c_dir)
        c[valid] = c_dir[valid] / norms[valid, np.newaxis]
        
        # Determine correct orientation: c should be "inside" the triangle.
        # Check dot product with centroid.
        centroids = (p1 + p2 + p3) / 3.0
        c_norms = np.linalg.norm(centroids, axis=1)
        valid_centroids = c_norms > 1e-10
        centroids[valid_centroids] /= c_norms[valid_centroids, np.newaxis]
        
        dots = np.sum(c * centroids, axis=1)
        c[dots < 0] *= -1
        
        # Compute angular distance from c to p1
        cos_theta = np.sum(c * p1, axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        # Only consider valid facets
        theta[~valid] = 0.0
        
        return np.max(theta) * 180.0 / np.pi
        
    except Exception:
        return 180.0

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    return points / norms

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _normalize(points)
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    current_radius = best_radius
    
    # Simulated Annealing parameters
    T_start = 1.0  # Initial temperature (in degrees? or rad? Let's use degrees for intuition, but exp needs consistent units)
    T_end = 0.001
    # We will run for a fixed number of steps or until time limit.
    # Let's estimate steps. _covering_radius takes ~5-10ms for N=130? 
    # If 10ms, we can do ~14000 steps in 140s.
    # Let's try 20000 steps, but check time frequently.
    
    steps = 0
    total_steps = 20000
    
    # To make SA effective, we need a step size that is comparable to the typical distance between points.
    # Typical distance ~ sqrt(4pi/N) ~ 0.3 rad ~ 17 deg.
    # Perturbation should be smaller, maybe 1-5 degrees initially, shrinking later.
    # However, covering radius is ~11 deg. So perturbations of ~1 deg are reasonable.
    
    while steps < total_steps:
        if time.time() > deadline:
            break
            
        # Cooling schedule
        # Linear cooling is simple.
        progress = steps / total_steps
        T = T_start * (T_end / T_start) ** progress
        
        # Pick a random point
        idx = np.random.randint(N)
        
        # Generate a random perturbation
        # Small random vector, added to current point, then normalized.
        # Step size should decrease with T? Or fixed?
        # Let's scale step size with T to explore more at high T.
        # Base step ~ 0.1 rad (approx 5 deg)
        step_scale = 0.1 * (T / T_start + 0.1) # Minimum step 0.01 rad
        
        # Random direction on sphere
        rand_vec = np.random.randn(3)
        rand_vec /= np.linalg.norm(rand_vec)
        
        # New position
        new_point = points[idx] + step_scale * rand_vec
        new_point /= np.linalg.norm(new_point)
        
        # Apply move
        old_point = points[idx].copy()
        points[idx] = new_point
        
        # Calculate new radius
        new_radius = _covering_radius(points)
        
        # Acceptance criterion
        delta = new_radius - current_radius
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            # Accept
            current_radius = new_radius
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
        else:
            # Reject
            points[idx] = old_point
        
        steps += 1
        
        # Check time periodically to avoid overshooting
        if steps % 100 == 0:
            if time.time() > deadline - 5.0:
                break

    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Covering radius: {_covering_radius(pts)} degrees")
    print(pts.shape)
