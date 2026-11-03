"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130
import numpy as np
import time
import scipy.spatial
import sys

# EVOLVE-BLOCK-START

def _compute_facet_radii(points):
    """
    Compute the circumradii of the convex hull facets.
    Returns:
        radii: array of circumradii for each facet
        facets: array of indices of points forming each facet
        circumcenters: array of circumcenters for each facet
    """
    try:
        hull = scipy.spatial.ConvexHull(points)
    except Exception:
        # Fallback if hull fails (e.g., degenerate points)
        return np.zeros(0), np.zeros((0, 3), dtype=int), np.zeros((0, 3))
    
    facets = hull.simplices
    radii = np.zeros(len(facets))
    circumcenters = np.zeros((len(facets), 3))
    
    for i, facet in enumerate(facets):
        v1 = points[facet[0]]
        v2 = points[facet[1]]
        v3 = points[facet[2]]
        
        # Compute circumcenter of triangle v1, v2, v3
        # Using the formula for circumcenter of a triangle in 3D
        a = v1 - v2
        b = v1 - v3
        c = v2 - v3
        
        # Normal vector to the plane
        n = np.cross(a, b)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            radii[i] = np.inf
            continue
        n = n / norm_n
        
        # Circumcenter formula
        # C = v1 + ( (|v1-v2|^2 * (v1-v3) x n + |v1-v3|^2 * n x (v1-v2) ) / (2 * |n|^2 * |n|^2) ) ... wait, simpler:
        # Solve for C such that |C-v1|^2 = |C-v2|^2 = |C-v3|^2
        # C lies in the plane defined by v1, v2, v3.
        # C = v1 + s * (v2-v1) + t * (v3-v1)
        # This is equivalent to finding the intersection of the plane and the line perpendicular to the plane through the centroid? No.
        
        # Use analytic solution:
        # d = 2 * (v1 x (v2 x v3)) + (v2 x v3) x (v1 x v2) ... complicated.
        # Simpler:
        # A = v2 - v1
        # B = v3 - v1
        # N = cross(A, B)
        # C = v1 + ( (dot(v1-v2, v1-v2) * cross(v1-v3, N) + dot(v1-v3, v1-v3) * cross(N, v1-v2)) / (2 * dot(N, N)) )
        
        v1_m_v2 = v1 - v2
        v1_m_v3 = v1 - v3
        N = np.cross(v2 - v1, v3 - v1)
        denom = 2.0 * np.dot(N, N)
        if denom < 1e-10:
            radii[i] = np.inf
            continue
            
        term1 = np.dot(v1_m_v2, v1_m_v2) * np.cross(v1_m_v3, N)
        term2 = np.dot(v1_m_v3, v1_m_v3) * np.cross(N, v1_m_v2)
        
        C = v1 + (term1 + term2) / denom
        
        circumcenters[i] = C
        R = np.linalg.norm(C - v1)
        radii[i] = R
        
    return radii, facets, circumcenters

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = np.inf
    
    # Calculate initial covering radius
    radii, facets, centers = _compute_facet_radii(points)
    if len(radii) > 0:
        curr_radius = np.max(radii)
    else:
        curr_radius = np.inf
    best_radius = curr_radius
    
    # Optimization parameters
    beta = 50.0  # Sharpness of log-sum-exp approximation
    eta = 0.01   # Initial step size
    min_eta = 1e-6
    decay = 0.99
    
    while time.time() < deadline:
        # Compute facet data
        radii, facets, centers = _compute_facet_radii(points)
        if len(radii) == 0:
            break
            
        curr_radius = np.max(radii)
        
        # Update best
        if curr_radius < best_radius:
            best_radius = curr_radius
            best_points = points.copy()
            
        # Compute gradient of smooth surrogate f = sum(exp(beta * R_k))
        # grad f = sum( beta * exp(beta * R_k) * grad R_k )
        # grad R_k w.r.t v_i is proj_{T_vi S^2} (C_k - v_i) / R_k
        
        grad = np.zeros_like(points)
        weights = beta * np.exp(beta * radii)
        
        for k in range(len(facets)):
            R_k = radii[k]
            if R_k < 1e-10 or R_k > np.pi:
                continue
            C_k = centers[k]
            w_k = weights[k]
            if w_k > 1e10: # Avoid overflow
                w_k = 1e10
                
            for idx in facets[k]:
                v_i = points[idx]
                # Vector from v_i to C_k
                vec = C_k - v_i
                # Project onto tangent space at v_i: vec - (vec . v_i) v_i
                proj = vec - np.dot(vec, v_i) * v_i
                # Gradient of R_k w.r.t v_i is proj / R_k
                grad[idx] += w_k * (proj / R_k)
        
        # Normalize gradient magnitude for stability
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad = grad / grad_norm
            
        # Update points
        # Riemannian gradient step
        new_points = points + eta * grad
        # Normalize back to sphere
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        new_points = new_points / norms
        
        # Check if this step improved the actual covering radius
        new_radii, _, _ = _compute_facet_radii(new_points)
        if len(new_radii) > 0:
            new_radius = np.max(new_radii)
        else:
            new_radius = np.inf
            
        if new_radius < curr_radius:
            points = new_points
            # Keep eta or increase slightly if we want to explore, but decreasing is safer
            eta *= 1.05 
            if eta > 0.1:
                eta = 0.1
        else:
            # Reject step, decrease eta
            eta *= decay
            if eta < min_eta:
                # Try a random restart or perturbation if stuck
                # Simple perturbation
                noise = np.random.normal(0, 0.01, points.shape)
                perturbed = points + noise
                perturbed /= np.linalg.norm(perturbed, axis=1, keepdims=True)
                p_radii, _, _ = _compute_facet_radii(perturbed)
                if len(p_radii) > 0 and np.max(p_radii) < curr_radius:
                    points = perturbed
                else:
                    # If stuck for too long, break or reset eta
                    pass
                    
    # Final check
    final_radii, _, _ = _compute_facet_radii(best_points)
    if len(final_radii) > 0:
        # Ensure no coincident points etc.
        return best_points
    else:
        return points

# EVOLVE-BLOCK-END
