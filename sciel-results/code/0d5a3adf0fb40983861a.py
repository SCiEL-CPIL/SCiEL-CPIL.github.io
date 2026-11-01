"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull
import warnings

N = 130

# EVOLVE-BLOCK-START

def _covering_radius(points):
    """Compute the covering radius in degrees using convex hull facets."""
    points = np.asarray(points, dtype=float)
    norms = np.linalg.norm(points, axis=1)
    norms[norms < 1e-12] = 1e-12
    points = points / norms[:, None]
    
    try:
        hull = ConvexHull(points)
        faces = hull.simplices
    except Exception:
        return _covering_radius_brute(points)
    
    # Vectorized circumradius calculation
    p1 = points[faces[:, 0]]
    p2 = points[faces[:, 1]]
    p3 = points[faces[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    n_vecs = np.cross(v1, v2)
    n_norms = np.linalg.norm(n_vecs, axis=1)
    
    valid = n_norms > 1e-12
    if not np.any(valid):
        return 0.0
        
    n_vecs[valid] /= n_norms[valid, None]
    
    dots = np.sum(n_vecs[valid] * p1[valid], axis=1)
    dots = np.clip(dots, -1, 1)
    
    # The circumradius is the angle between the normal and the vertex.
    # We take the minimum of arccos(dot) and arccos(-dot) to handle orientation?
    # Actually, for a convex hull, the normal points outward.
    # The circumcenter is in the direction of the normal (or opposite).
    # The distance is arccos(|n . p1|) ? No.
    # If n . p1 > 0, the center is "above" the triangle.
    # The radius is arccos(n . p1).
    # If n . p1 < 0, the center is "below".
    # The radius is arccos(-n . p1).
    # So radius = arccos(|n . p1|).
    radii = np.arccos(np.abs(dots))
    max_circ = np.max(radii)
    
    return np.degrees(max_circ)


def _covering_radius_brute(points, n_theta=180, n_phi=360):
    """Brute force covering radius computation on a grid."""
    points = np.asarray(points, dtype=float)
    norms = np.linalg.norm(points, axis=1)
    norms[norms < 1e-12] = 1e-12
    points = points / norms[:, None]
    
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    max_dist = 0.0
    chunk_size = 10000
    for start in range(0, len(grid), chunk_size):
        end = min(start + chunk_size, len(grid))
        g = grid[start:end]
        dots = g @ points.T
        dots = np.clip(dots, -1, 1)
        max_dots = np.max(dots, axis=1)
        min_angs = np.arccos(max_dots)
        max_dist = max(max_dist, np.max(min_angs))
    
    return np.degrees(max_dist)


def _lloyd_iteration(points, n_iter=20, tol=1e-10):
    """Lloyd's algorithm for spherical k-means / Voronoi centroid refinement."""
    points = np.array(points, dtype=float)
    n = len(points)
    
    for it in range(n_iter):
        # Use a coarser grid for speed, as Polish will refine
        n_theta = 180
        n_phi = 360
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        TH, PH = np.meshgrid(theta, phi, indexing='ij')
        x = np.sin(TH) * np.cos(PH)
        y = np.sin(TH) * np.sin(PH)
        z = np.cos(TH)
        grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        
        # Assign each grid point to nearest site
        dots = grid @ points.T  # (len(grid), n)
        nearest = np.argmax(dots, axis=1)
        
        # Compute centroids of each Voronoi cell
        new_points = np.zeros_like(points)
        counts = np.zeros(n)
        for i in range(n):
            mask = nearest == i
            if np.sum(mask) > 0:
                new_points[i] = grid[mask].mean(axis=0)
                counts[i] = np.sum(mask)
        
        # Handle empty cells: keep original point
        for i in range(n):
            if counts[i] == 0:
                new_points[i] = points[i]
        
        # Normalize
        norms = np.linalg.norm(new_points, axis=1)
        norms[norms < 1e-12] = 1e-12
        new_points = new_points / norms[:, None]
        
        # Check convergence
        diff = np.linalg.norm(new_points - points)
        points = new_points
        if diff < tol:
            break
    
    return points


def _facet_polish(points, max_time=30.0):
    """Polish by iteratively moving worst-facet vertices toward their circumcenter."""
    points = np.array(points, dtype=float)
    start = time.time()
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    lr = 0.1
    no_improve = 0
    K = 10 # Top K worst facets
    
    while time.time() - start < max_time and no_improve < 100:
        norms = np.linalg.norm(points, axis=1)
        norms[norms < 1e-12] = 1e-12
        points = points / norms[:, None]
        
        try:
            hull = ConvexHull(points)
            faces = hull.simplices
        except Exception:
            break
        
        # Vectorized circumradius calculation
        p1 = points[faces[:, 0]]
        p2 = points[faces[:, 1]]
        p3 = points[faces[:, 2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        n_vecs = np.cross(v1, v2)
        n_norms = np.linalg.norm(n_vecs, axis=1)
        
        valid = n_norms > 1e-12
        if not np.any(valid):
            break
            
        n_vecs[valid] /= n_norms[valid, None]
        
        dots = np.sum(n_vecs[valid] * p1[valid], axis=1)
        dots = np.clip(dots, -1, 1)
        radii = np.arccos(np.abs(dots))
        
        # Find top K worst facets
        valid_indices = np.where(valid)[0]
        if len(radii) < K:
            top_k_worst_indices = valid_indices
        else:
            top_k_global = np.argpartition(radii, -K)[-K:]
            top_k_worst_indices = valid_indices[top_k_global]
            
        worst_faces = faces[top_k_worst_indices]
        unique_verts = np.unique(worst_faces)
        
        # Move these vertices towards their local circumcenters
        new_points = points.copy()
        for idx in unique_verts:
            # Find which of the worst facets contain idx
            masks = (worst_faces == idx).any(axis=1)
            if not np.any(masks): continue
            relevant_facets = worst_faces[masks]
            
            r_p1 = points[relevant_facets[:, 0]]
            r_p2 = points[relevant_facets[:, 1]]
            r_p3 = points[relevant_facets[:, 2]]
            
            rv1 = r_p2 - r_p1
            rv2 = r_p3 - r_p1
            rn = np.cross(rv1, rv2)
            rn_norms = np.linalg.norm(rn, axis=1)
            valid_r = rn_norms > 1e-12
            if not np.any(valid_r): continue
            
            rn[valid_r] /= rn_norms[valid_r, None]
            
            # Circumcenter is the normalized normal
            centers = rn[valid_r]
            avg_center = np.sum(centers, axis=0)
            avg_norm = np.linalg.norm(avg_center)
            if avg_norm > 1e-12:
                avg_center /= avg_norm
                p = points[idx]
                d = avg_center - p
                d = d - (d @ p) * p # Project to tangent plane
                d_norm = np.linalg.norm(d)
                if d_norm > 1e-12:
                    d /= d_norm
                    new_points[idx] = p + lr * d
        
        # Normalize
        norms_t = np.linalg.norm(new_points, axis=1)
        norms_t[norms_t < 1e-12] = 1e-12
        new_points = new_points / norms_t[:, None]
        
        new_radius = _covering_radius(new_points)
        
        if new_radius < best_radius - 1e-8:
            best_radius = new_radius
            best_points = new_points.copy()
            points = new_points.copy()
            no_improve = 0
        else:
            no_improve += 1
            lr *= 0.9
            if lr < 0.005:
                break
            points = best_points.copy()
                
    return best_points


def _fibonacci_spiral(n):
    """Generate points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    norms = np.linalg.norm(points, axis=1)
    return points / norms[:, None]


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Stage 1: Fibonacci spiral initialization
    points = _fibonacci_spiral(N)
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Stage 2: Lloyd's algorithm for global balance (faster, coarser grid)
    if time.time() < deadline - 60:
        lloyd_points = _lloyd_iteration(points, n_iter=20)
        lloyd_radius = _covering_radius(lloyd_points)
        if lloyd_radius < best_radius:
            best_points = lloyd_points
            best_radius = lloyd_radius
    
    # Stage 3: Facet-based polishing (vectorized, aggressive)
    remaining = deadline - time.time() - 5
    if remaining > 10:
        polished = _facet_polish(best_points, max_time=remaining)
        polish_radius = _covering_radius(polished)
        if polish_radius < best_radius:
            best_points = polished
            best_radius = polish_radius
            
    # Stage 4: Final quick polish if time permits
    remaining = deadline - time.time() - 2
    if remaining > 5:
        polished2 = _facet_polish(best_points, max_time=remaining)
        p2_radius = _covering_radius(polished2)
        if p2_radius < best_radius:
            best_points = polished2
            best_radius = p2_radius

    # Final normalization
    norms = np.linalg.norm(best_points, axis=1)
    norms[norms < 1e-12] = 1e-12
    best_points = best_points / norms[:, None]
    
    return best_points

# EVOLVE-BLOCK-END
