import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def construct():
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    start_time = time.time()
    time_limit = 145.0 # Safety margin for 150s limit
    
    best_points = points.copy()
    best_radius = np.inf
    
    def get_covering_radius(pts):
        try:
            hull = ConvexHull(pts)
            max_rad = 0.0
            for simplex in hull.simplices:
                facet_pts = pts[simplex]
                v0, v1, v2 = facet_pts[0], facet_pts[1], facet_pts[2]
                n = np.cross(v1 - v0, v2 - v0)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-10:
                    continue
                n /= n_norm
                # Ensure normal points outward
                centroid = (v0 + v1 + v2) / 3.0
                if n @ centroid < 0:
                    n = -n
                dist = np.arccos(np.clip(n @ v0, -1, 1))
                if dist > max_rad:
                    max_rad = dist
            return max_rad
        except Exception:
            return np.pi

    def compute_forces(pts):
        """
        Computes force on each point to reduce covering radius.
        Force is sum of attraction to facet circumcenters weighted by facet radius squared,
        plus small repulsion to prevent coincident points.
        """
        forces = np.zeros_like(pts)
        try:
            hull = ConvexHull(pts)
            for simplex in hull.simplices:
                facet_pts = pts[simplex]
                v0, v1, v2 = facet_pts[0], facet_pts[1], facet_pts[2]
                n = np.cross(v1 - v0, v2 - v0)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-10:
                    continue
                n /= n_norm
                centroid = (v0 + v1 + v2) / 3.0
                if n @ centroid < 0:
                    n = -n
                dist = np.arccos(np.clip(n @ v0, -1, 1))
                weight = dist * dist
                
                # Apply force to the 3 vertices
                for idx in simplex:
                    forces[idx] += weight * (n - pts[idx])
            
        except Exception:
            pass
        
        # Add repulsion
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        inv_dist_sq2 = 1.0 / (dist_sq * dist_sq)
        np.fill_diagonal(inv_dist_sq2, 0.0)
        
        repulsion = np.sum(diff * inv_dist_sq2[:, :, np.newaxis], axis=1)
        
        rep_scale = 1e-4
        forces += rep_scale * repulsion
        
        return forces

    # Main Loop
    eta = 0.01
    while time.time() - start_time < time_limit:
        forces = compute_forces(points)
        
        # Update
        points += eta * forces
        
        # Normalize
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Check radius
        current_rad = get_covering_radius(points)
        
        if current_rad < best_radius:
            best_radius = current_rad
            best_points = points.copy()
            eta *= 0.999
        else:
            eta *= 0.95
            if eta < 1e-6:
                break

    # Final check
    best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
    return best_points
