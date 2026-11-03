"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time


def _covering_radius(points):
    """Compute the covering radius (in radians) of points on the unit sphere."""
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback if hull fails (should be rare for well-distributed points)
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        n_theta = 200
        n_phi = 400
        thetas = np.linspace(0, np.pi, n_theta)
        phis = np.linspace(0, 2 * np.pi, n_phi)
        T, P = np.meshgrid(thetas, phis, indexing='ij')
        test_pts = np.column_stack([
            np.sin(T) * np.cos(P).ravel(),
            np.sin(T) * np.sin(P).ravel(),
            np.cos(T).ravel()
        ])
        dists, _ = tree.query(test_pts)
        max_chord = np.max(dists)
        max_angle = 2 * np.arcsin(min(max_chord / 2, 1.0))
        return max_angle
    
    max_radius = 0.0
    for simplex in hull.simplices:
        facet_pts = points[simplex]
        a, b, c = facet_pts
        
        # Normal vector to the plane
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        if norm_cross < 1e-15:
            continue
        
        # Circumcenter on the sphere
        o = cross / norm_cross
        
        # Ensure o is the correct pole (facing outwards)
        # The hull normal points outwards. 
        # Check dot product with the centroid of the triangle
        centroid = (a + b + c) / 3.0
        if np.dot(o, centroid) < 0:
            o = -o
            
        cos_angle = np.clip(np.dot(o, a), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        max_radius = max(max_radius, angle)
    
    return max_radius


def _refine_covering_gradient(points, n_iterations=5000, lr_init=0.05, lr_min=0.001):
    """
    Refine points using gradient descent on the covering radius.
    Moves vertices towards the circumcenters of their adjacent facets.
    """
    points = points.copy()
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    for it in range(n_iterations):
        try:
            hull = ConvexHull(points)
        except Exception:
            # If hull fails, perturb slightly and retry or break
            points += 0.01 * np.random.randn(*points.shape)
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            continue
            
        forces = np.zeros_like(points)
        max_r = 0.0
        facet_radii = []
        
        # First pass: compute radii to determine weights
        for simplex in hull.simplices:
            a_idx, b_idx, c_idx = simplex
            a, b, c = points[a_idx], points[b_idx], points[c_idx]
            
            v1 = a - b
            v2 = a - c
            cross = np.cross(v1, v2)
            norm_cross = np.linalg.norm(cross)
            if norm_cross < 1e-15:
                facet_radii.append(0.0)
                continue
                
            o = cross / norm_cross
            centroid = (a + b + c) / 3.0
            if np.dot(o, centroid) < 0:
                o = -o
                
            cos_angle = np.clip(np.dot(o, a), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            facet_radii.append(angle)
            max_r = max(max_r, angle)
            
        # Second pass: accumulate forces
        # We only apply forces from facets that are close to the maximum radius
        # to avoid moving points that are in "good" regions.
        threshold = max_r * 0.95 if max_r > 0 else 1.0
        
        for i, simplex in enumerate(hull.simplices):
            r = facet_radii[i]
            if r < threshold:
                continue
                
            a_idx, b_idx, c_idx = simplex
            a, b, c = points[a_idx], points[b_idx], points[c_idx]
            
            v1 = a - b
            v2 = a - c
            cross = np.cross(v1, v2)
            norm_cross = np.linalg.norm(cross)
            if norm_cross < 1e-15:
                continue
                
            o = cross / norm_cross
            centroid = (a + b + c) / 3.0
            if np.dot(o, centroid) < 0:
                o = -o
                
            # Force direction: towards circumcenter o
            # Project (o - point) onto tangent plane at point
            # Tangent plane normal is the point itself.
            # Proj = (o - p) - ((o - p) . p) p
            
            for idx, p in zip([a_idx, b_idx, c_idx], [a, b, c]):
                vec = o - p
                proj = vec - np.dot(vec, p) * p
                # Weight by how close r is to max_r
                weight = (r / max_r) ** 2 if max_r > 0 else 1.0
                forces[idx] += weight * proj

        # Normalize forces
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms = np.maximum(force_norms, 1e-15)
        force_unit = forces / force_norms
        
        # Learning rate decay
        progress = it / n_iterations
        lr = lr_init * (lr_min / lr_init) ** progress
        
        # Update
        points = points + lr * force_unit
        
        # Project to sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Check improvement
        if it % 100 == 0:
            current_radius = _covering_radius(points)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
                
    return best_points, best_radius


def _fibonacci_spiral(n, offset=0.5):
    """Generate points using Fibonacci spiral."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Generate random points on the unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    rng = np.random.default_rng(42)
    
    best_points = None
    best_radius = np.inf
    
    # Initialization 1: Fibonacci spiral
    points1 = _fibonacci_spiral(N)
    # Add small noise to break symmetry
    points1 = points1 + 0.01 * rng.standard_normal((N, 3))
    norms = np.linalg.norm(points1, axis=1, keepdims=True)
    points1 = points1 / norms
    
    points1, radius1 = _refine_covering_gradient(points1, n_iterations=3000, lr_init=0.05)
    
    if radius1 < best_radius:
        best_radius = radius1
        best_points = points1.copy()
        
    # Initialization 2: Random
    points2 = _random_sphere(N, rng)
    points2, radius2 = _refine_covering_gradient(points2, n_iterations=3000, lr_init=0.05)
    
    if radius2 < best_radius:
        best_radius = radius2
        best_points = points2.copy()

    # Initialization 3: Fibonacci with different offset
    points3 = _fibonacci_spiral(N, offset=0.3)
    points3 = points3 + 0.01 * rng.standard_normal((N, 3))
    norms = np.linalg.norm(points3, axis=1, keepdims=True)
    points3 = points3 / norms
    points3, radius3 = _refine_covering_gradient(points3, n_iterations=3000, lr_init=0.05)
    
    if radius3 < best_radius:
        best_radius = radius3
        best_points = points3.copy()

    # Final fine-tuning on the best configuration
    if best_points is not None:
        best_points, final_radius = _refine_covering_gradient(best_points, n_iterations=2000, lr_init=0.01, lr_min=0.0001)
        if final_radius < best_radius:
            best_radius = final_radius
            # best_points is already updated in the function return

    # Ensure points are on the unit sphere
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points
# EVOLVE-BLOCK-END
