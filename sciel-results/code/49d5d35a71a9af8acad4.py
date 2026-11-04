"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import random

def _covering_radius(pts):
    """Compute the covering radius in degrees for points on the unit sphere."""
    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    pts = pts / norms
    
    # Check if all points are in a closed hemisphere
    # If they are, the covering radius is 180 degrees (or close to it)
    # A simple check: if there's a point p such that all points have positive dot product with p
    # Actually, a better check: the origin is not in the convex hull if they're in a hemisphere
    # For simplicity, we'll check if the maximum dot product with any direction is > 0
    # But this is expensive. Let's just compute the convex hull and check.
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
    
    # For each facet of the convex hull, find the circumradius of that facet
    # The covering radius is the maximum circumradius over all facets
    # A facet is a set of points that form a face of the convex hull
    
    max_circumradius = 0.0
    
    for simplex in hull.simplices:
        # Get the points of this facet
        facet_pts = pts[simplex]
        
        # For a spherical cap defined by these points, the circumradius is the angular distance
        # from the center of the cap to the farthest point
        # The center of the cap is in the direction of the normal to the facet
        
        # Compute the normal to the facet
        if len(simplex) == 3:
            # Triangle
            v1 = facet_pts[1] - facet_pts[0]
            v2 = facet_pts[2] - facet_pts[0]
            normal = np.cross(v1, v2)
        elif len(simplex) == 2:
            # Edge (shouldn't happen for 3D convex hull, but just in case)
            v1 = facet_pts[1] - facet_pts[0]
            # Normal is perpendicular to v1 and the midpoint direction
            mid = (facet_pts[0] + facet_pts[1]) / 2.0
            normal = np.cross(v1, mid)
        else:
            # Higher dimensional simplex
            # Use the first 3 points
            v1 = facet_pts[1] - facet_pts[0]
            v2 = facet_pts[2] - facet_pts[0]
            normal = np.cross(v1, v2)
        
        norm_norm = np.linalg.norm(normal)
        if norm_norm < 1e-10:
            continue
        normal = normal / norm_norm
        
        # The center of the spherical cap is in the direction of the normal
        # (or -normal, depending on orientation)
        # The angular distance from the center to each facet point is the circumradius
        # We want the maximum angular distance
        
        # The center should be on the same side as the origin relative to the facet
        # For a convex hull containing the origin, the normal should point away from the origin
        # Let's check the dot product with the origin (which is 0)
        # Actually, for a convex hull of points on the unit sphere, the origin is inside
        # if and only if the points are not in a closed hemisphere
        
        # The circumradius is the maximum angular distance from the normal direction to any facet point
        cos_angles = np.dot(facet_pts, normal)
        # Clamp to [-1, 1] for numerical stability
        cos_angles = np.clip(cos_angles, -1.0, 1.0)
        angles = np.arccos(cos_angles)
        
        circumradius = np.max(angles)
        if circumradius > max_circumradius:
            max_circumradius = circumradius
    
    return np.degrees(max_circumradius)

def _random_sphere_points(n, seed=None):
    """Generate n random points on the unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    # Use Marsaglia's method for uniform distribution on the sphere
    pts = np.zeros((n, 3))
    for i in range(n):
        while True:
            x = rng.uniform(-1, 1)
            y = rng.uniform(-1, 1)
            r2 = x * x + y * y
            if r2 < 1.0 and r2 > 0:
                z = np.sqrt(1.0 - r2)
                pts[i] = [x, y, z]
                break
    return pts

def _refine(pts, max_iters=50, tol=1e-8):
    """Refine points using gradient descent on the covering radius."""
    pts = pts.copy()
    best_pts = pts.copy()
    best_radius = _covering_radius(pts)
    
    for it in range(max_iters):
        # Compute the current covering radius and identify the worst facet
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        max_circumradius = 0.0
        worst_facet = None
        worst_normal = None
        
        for simplex in hull.simplices:
            facet_pts = pts[simplex]
            
            if len(simplex) == 3:
                v1 = facet_pts[1] - facet_pts[0]
                v2 = facet_pts[2] - facet_pts[0]
                normal = np.cross(v1, v2)
            elif len(simplex) == 2:
                v1 = facet_pts[1] - facet_pts[0]
                mid = (facet_pts[0] + facet_pts[1]) / 2.0
                normal = np.cross(v1, mid)
            else:
                v1 = facet_pts[1] - facet_pts[0]
                v2 = facet_pts[2] - facet_pts[0]
                normal = np.cross(v1, v2)
            
            norm_norm = np.linalg.norm(normal)
            if norm_norm < 1e-10:
                continue
            normal = normal / norm_norm
            
            cos_angles = np.dot(facet_pts, normal)
            cos_angles = np.clip(cos_angles, -1.0, 1.0)
            angles = np.arccos(cos_angles)
            
            circumradius = np.max(angles)
            if circumradius > max_circumradius:
                max_circumradius = circumradius
                worst_facet = simplex
                worst_normal = normal
        
        if worst_facet is None:
            break
        
        # Move the points in the worst facet slightly towards the normal direction
        # to reduce the circumradius
        step_size = 0.01
        for idx in worst_facet:
            pts[idx] += step_size * worst_normal
            # Renormalize
            norm = np.linalg.norm(pts[idx])
            if norm > 0:
                pts[idx] /= norm
        
        new_radius = _covering_radius(pts)
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = pts.copy()
        else:
            # Revert
            pts = best_pts.copy()
            step_size *= 0.5
            if step_size < 1e-6:
                break
    
    return best_pts, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_pts = None
    best_radius = 180.0
    
    # Start with a Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    current_radius = _covering_radius(pts)
    if current_radius < best_radius:
        best_radius = current_radius
        best_pts = pts.copy()
    
    # Try multiple random starts
    for trial in range(20):
        if time.time() > deadline:
            break
        
        seed = 42 + trial
        random_pts = _random_sphere_points(N, seed=seed)
        
        # Refine the random points
        refined_pts, radius = _refine(random_pts, max_iters=30)
        
        if radius < best_radius:
            best_radius = radius
            best_pts = refined_pts.copy()
        
        # Also try refining the Fibonacci spiral with random perturbations
        if time.time() > deadline:
            break
        
        perturbed = pts.copy()
        rng = np.random.RandomState(seed)
        for _ in range(5):
            # Perturb a few random points
            idxs = rng.choice(N, size=10, replace=False)
            for idx in idxs:
                # Move the point slightly in a random direction
                random_dir = rng.normal(size=3)
                random_dir /= np.linalg.norm(random_dir)
                perturbed[idx] += 0.05 * random_dir
                norm = np.linalg.norm(perturbed[idx])
                if norm > 0:
                    perturbed[idx] /= norm
            
            refined_pts, radius = _refine(perturbed, max_iters=20)
            if radius < best_radius:
                best_radius = radius
                best_pts = refined_pts.copy()
    
    # Final refinement of the best points
    if time.time() < deadline - 10.0:
        final_pts, final_radius = _refine(best_pts, max_iters=100)
        if final_radius < best_radius:
            best_radius = final_radius
            best_pts = final_pts
    
    # Ensure all points are normalized and non-zero
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    best_pts = best_pts / norms
    
    return best_pts
# EVOLVE-BLOCK-END
