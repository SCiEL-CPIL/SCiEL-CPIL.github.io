"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull

def _compute_facet_data(points):
    """
    Compute circumradii and circumcenters for all convex hull facets vectorized.
    Returns:
    radii: array of shape (M,)
    centers: array of shape (M, 3)
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.array([np.pi]), np.array([np.zeros(3)])

    simplices = hull.simplices
    p1 = points[simplices[:, 0]]
    p2 = points[simplices[:, 1]]
    p3 = points[simplices[:, 2]]

    v = p2 - p1
    w = p3 - p1

    vv = np.sum(v * v, axis=1)
    ww = np.sum(w * w, axis=1)
    vw = np.sum(v * w, axis=1)

    det = vv * ww - vw * vw
    
    # Avoid division by zero
    valid = det > 1e-12
    det_safe = np.where(valid, det, 1.0)

    t1 = (0.5 * vv * ww - 0.5 * ww * vw) / det_safe
    t2 = (0.5 * ww * vv - 0.5 * vv * vw) / det_safe

    C_plane = p1 + t1[:, np.newaxis] * v + t2[:, np.newaxis] * w
    
    norm_C = np.linalg.norm(C_plane, axis=1)
    valid &= norm_C > 1e-12
    norm_C_safe = np.where(valid, norm_C, 1.0)
    
    centers = C_plane / norm_C_safe[:, np.newaxis]
    
    cos_R = np.sum(centers * p1, axis=1)
    cos_R = np.clip(cos_R, -1.0, 1.0)
    radii = np.arccos(cos_R)
    
    # Set invalid facets to large radius
    radii = np.where(valid, radii, np.pi)
    
    return radii, centers

def _covering_radius(points):
    """Compute the covering radius (in radians)."""
    radii, _ = _compute_facet_data(points)
    return np.max(radii)

def _optimize(points, iterations=5000):
    """
    Geometric relaxation: Move vertices of largest facets towards circumcenters.
    """
    points = points.copy()
    best_points = points.copy()
    best_energy = _covering_radius(points)
    
    for it in range(iterations):
        radii, centers = _compute_facet_data(points)
        
        if np.max(radii) < 0.197: # ~11.3 degrees
            break
            
        # Select top K facets to update to balance load
        # K=1 is slow, K=10 is faster convergence
        K = 10
        if len(radii) < K:
            K = len(radii)
            
        top_k_indices = np.argsort(radii)[-K:]
        
        # Get the vertices involved in these facets
        # We need to map facet indices to point indices
        # Since we don't have simplices returned, we need to re-fetch or store them
        # Let's modify _compute_facet_data to return simplices too for efficiency in loop?
        # Or just re-compute hull? No, that's expensive.
        # Let's assume we can just move the points corresponding to the max radius facets.
        # But we need the indices.
        
        # For simplicity and speed, let's just move the single max facet first, 
        # but vectorized logic for top-K is better.
        # Let's just pick the single max for now to keep code simple, 
        # but do it in a loop of small steps.
        
        max_idx = np.argmax(radii)
        max_r = radii[max_idx]
        
        # We need the simplex for max_idx. 
        # Since _compute_facet_data doesn't return simplices, we have a slight inefficiency.
        # Let's refactor to return simplices.
        
        # Actually, let's just use the centers. 
        # The center is the target. 
        # But which points to move? The ones defining the facet.
        
        # To avoid refactoring the return signature drastically in a small edit,
        # let's just re-compute the hull and get the simplex for the max index.
        # This is acceptable if N is small (130).
        
        try:
            hull = ConvexHull(points)
            simplex = hull.simplices[max_idx]
        except:
            break
            
        C = centers[max_idx]
        
        # Adaptive step size
        step = 0.01 * (max_r / np.pi) + 0.001
        
        for idx in simplex:
            p = points[idx]
            new_p = p + step * (C - p)
            new_p = new_p / np.linalg.norm(new_p)
            points[idx] = new_p
            
        energy = _covering_radius(points)
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
            
    return best_points

def construct():
    # Deterministic Initialization: Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Optimize
    best_points = _optimize(points, iterations=5000)
    
    # Final normalization
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    return best_points
# EVOLVE-BLOCK-END
