"""Rewritten program for covering the sphere with 50 equal caps."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time


def _compute_circumcenters_and_radii(pts, hull):
    """
    Compute circumcenters and circumradii for all facets of the convex hull.
    Returns:
        circumcenters: array of shape (n_facets, 3)
        radii: array of shape (n_facets,) in radians
    """
    facets = hull.simplices
    n_facets = len(facets)
    circumcenters = np.zeros((n_facets, 3))
    radii = np.zeros(n_facets)
    
    for i, facet in enumerate(facets):
        p1 = pts[facet[0]]
        p2 = pts[facet[1]]
        p3 = pts[facet[2]]
        
        # Compute circumcenter of the spherical triangle
        # Formula: cc = (a*(b x c) + b*(c x a) + c*(a x b)) / |...| 
        # Actually, for unit vectors, the circumcenter is the normalized vector:
        # cc_dir = (1 - (b.c)^2)*a + (1 - (c.a)^2)*b + (1 - (a.b)^2)*c
        
        ab = np.dot(p1, p2)
        bc = np.dot(p2, p3)
        ca = np.dot(p3, p1)
        
        coeff1 = 1.0 - bc * bc
        coeff2 = 1.0 - ca * ca
        coeff3 = 1.0 - ab * ab
        
        cc_vec = coeff1 * p1 + coeff2 * p2 + coeff3 * p3
        cc_norm = np.linalg.norm(cc_vec)
        
        if cc_norm < 1e-10:
            # Degenerate facet, skip or assign zero radius
            radii[i] = 0.0
            circumcenters[i] = p1 # Arbitrary
            continue
            
        cc = cc_vec / cc_norm
        
        # Determine the correct side (outward normal direction)
        # The circumcenter should be in the same hemisphere as the facet normal
        n = np.cross(p2 - p1, p3 - p1)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            radii[i] = 0.0
            circumcenters[i] = p1
            continue
        n = n / n_norm
        
        if np.dot(cc, n) < 0:
            cc = -cc
            # If we flipped, the radius calculation remains the same (angle to vertex)
        
        circumcenters[i] = cc
        
        # Compute circumradius (angular distance from cc to any vertex)
        cos_theta = np.dot(cc, p1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        radii[i] = theta
        
    return circumcenters, radii


def _covering_radius_degrees(pts):
    """Compute the covering radius in degrees for points on the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 90.0
    
    cc, radii = _compute_circumcenters_and_radii(pts, hull)
    
    if len(radii) == 0:
        return 90.0
        
    max_rad = np.max(radii)
    return np.degrees(max_rad)


def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius
    using force dynamics towards facet circumcenters.
    """
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    best_pts = pts.copy()
    best_radius = _covering_radius_degrees(pts)
    
    # Parameters for dynamics
    num_steps = 3000
    base_step = 0.05
    
    for step in range(num_steps):
        # Decay step size
        step_size = base_step * (1.0 - step / num_steps)
        
        # Compute hull and forces
        try:
            hull = ConvexHull(pts)
        except Exception:
            # If hull fails, perturb randomly
            pts += 0.01 * np.random.randn(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            continue
            
        cc, radii = _compute_circumcenters_and_radii(pts, hull)
        
        forces = np.zeros_like(pts)
        
        # Accumulate forces: move towards circumcenter of each adjacent facet
        # Weighted by the radius (larger caps pull harder)
        for i, facet in enumerate(hull.simplices):
            r_facet = radii[i]
            if r_facet < 1e-10:
                continue
                
            c = cc[i]
            
            # Weight: prioritize reducing larger caps
            # Using r^2 or just r. Let's use r to be linear.
            weight = r_facet
            
            for idx in facet:
                p = pts[idx]
                # Vector from p to c
                vec = c - p
                # Project to tangent plane at p
                vec_tangent = vec - np.dot(vec, p) * p
                vec_norm = np.linalg.norm(vec_tangent)
                
                if vec_norm > 1e-10:
                    # Force direction is towards c
                    # Magnitude proportional to weight
                    forces[idx] += weight * (vec_tangent / vec_norm)
        
        # Update positions
        for i in range(N):
            if np.linalg.norm(forces[i]) > 1e-10:
                # Normalize force direction
                f_dir = forces[i] / np.linalg.norm(forces[i])
                # Move
                new_p = pts[i] + step_size * f_dir
                new_norm = np.linalg.norm(new_p)
                if new_norm > 1e-10:
                    pts[i] = new_p / new_norm
        
        # Check improvement periodically
        if step % 50 == 0:
            current_radius = _covering_radius_degrees(pts)
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = pts.copy()
        
        # Early stopping if we hit a very good solution
        if best_radius < 18.31:
             # We are close to the best known. Try to refine further with smaller steps?
             # Or just continue.
             pass

    # Final check
    final_radius = _covering_radius_degrees(pts)
    if final_radius < best_radius:
        best_pts = pts.copy()
    
    # Normalize final points
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    return best_pts
# EVOLVE-BLOCK-END
