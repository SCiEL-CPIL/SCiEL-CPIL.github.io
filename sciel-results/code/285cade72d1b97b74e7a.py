"""Rewritten program for covering the sphere with 50 equal caps."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _compute_circumcenters(pts, facets):
    """
    Compute circumcenters and circumradii (angular) for each facet.
    Returns arrays of centers (n_facets, 3) and radii (n_facets,).
    """
    n_facets = len(facets)
    centers = np.zeros((n_facets, 3))
    radii = np.zeros(n_facets)
    
    for idx, facet in enumerate(facets):
        p1, p2, p3 = pts[facet[0]], pts[facet[1]], pts[facet[2]]
        
        # Vector formula for circumcenter on unit sphere
        a = p1
        b = p2
        c = p3
        
        ab = np.dot(a, b)
        bc = np.dot(b, c)
        ca = np.dot(c, a)
        
        # Coefficients for the linear combination
        # cc = (1 - bc^2)a + (1 - ca^2)b + (1 - ab^2)c
        coeff_a = 1.0 - bc * bc
        coeff_b = 1.0 - ca * ca
        coeff_c = 1.0 - ab * ab
        
        cc_vec = coeff_a * a + coeff_b * b + coeff_c * c
        cc_norm = np.linalg.norm(cc_vec)
        
        if cc_norm < 1e-10:
            # Degenerate case, skip or handle
            centers[idx] = a
            radii[idx] = 0.0
            continue
            
        cc = cc_vec / cc_norm
        
        # Determine correct orientation
        # Normal of the facet
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            centers[idx] = a
            radii[idx] = 0.0
            continue
        n = n / n_norm
        
        # The circumcenter must be on the same side of the plane as the normal
        # (outward normal of convex hull)
        if np.dot(cc, n) < 0:
            cc = -cc
        
        centers[idx] = cc
        
        # Compute angular radius
        cos_theta = np.dot(cc, a)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        radii[idx] = theta
        
    return centers, radii


def _covering_radius_degrees(pts):
    """Compute the covering radius in degrees for points on the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 90.0
        
    facets = hull.simplices
    centers, radii = _compute_circumcenters(pts, facets)
    
    if len(radii) == 0:
        return 90.0
        
    return np.degrees(np.max(radii))


def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius
    using facet-circumcenter attraction dynamics.
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
    
    # Parameters for the dynamics
    num_steps = 3000
    base_lr = 0.05
    
    for step in range(num_steps):
        # Decay learning rate
        # Use a cosine annealing or linear decay
        lr = base_lr * (1.0 - step / num_steps)
        if lr < 1e-5:
            lr = 1e-5
            
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        facets = hull.simplices
        centers, radii = _compute_circumcenters(pts, facets)
        
        # Map each point to its incident facets
        # We need to find, for each point i, which facets contain it.
        # hull.simplices has shape (n_facets, 3)
        # We can create an array of size N x (max_facets_per_point) or just iterate.
        # Since N=50, simple iteration is fine.
        
        forces = np.zeros((N, 3))
        
        # For each point, find the worst facet
        # We can precompute a list of facet indices for each point
        point_facets = [[] for _ in range(N)]
        for f_idx, facet in enumerate(facets):
            for p_idx in facet:
                point_facets[p_idx].append(f_idx)
                
        for i in range(N):
            if not point_facets[i]:
                continue
                
            # Find the facet with the largest radius
            max_rad = -1
            worst_f_idx = -1
            for f_idx in point_facets[i]:
                if radii[f_idx] > max_rad:
                    max_rad = radii[f_idx]
                    worst_f_idx = f_idx
                    
            if worst_f_idx == -1:
                continue
                
            c = centers[worst_f_idx]
            p = pts[i]
            
            # Force direction: from p to c
            # Project onto tangent plane at p
            diff = c - p
            f_tangent = diff - np.dot(diff, p) * p
            
            f_norm = np.linalg.norm(f_tangent)
            if f_norm < 1e-10:
                continue
                
            # Scale force by the radius (larger holes get stronger pull)
            # Use radius in radians
            mag = max_rad 
            
            forces[i] = (f_tangent / f_norm) * mag
            
        # Update positions
        new_pts = pts.copy()
        for i in range(N):
            f = forces[i]
            if np.linalg.norm(f) < 1e-10:
                continue
                
            # Move
            new_pts[i] = pts[i] + lr * f
            
            # Renormalize
            n = np.linalg.norm(new_pts[i])
            if n > 1e-10:
                new_pts[i] = new_pts[i] / n
            else:
                new_pts[i] = pts[i]
                
        pts = new_pts
        
        # Check improvement periodically
        if step % 50 == 0:
            current_radius = _covering_radius_degrees(pts)
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = pts.copy()
                
    # Final check
    final_radius = _covering_radius_degrees(pts)
    if final_radius < best_radius:
        best_pts = pts.copy()
        
    # Normalize final points
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    return best_pts
# EVOLVE-BLOCK-END
