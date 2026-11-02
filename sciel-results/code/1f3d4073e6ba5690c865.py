import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Helper to calculate circumcenter and radius of a spherical triangle
    def get_circumcenter_and_radius(v1, v2, v3):
        # Calculate planar circumcenter
        # Using the formula:
        # C = v1 + alpha*(v2-v1) + beta*(v3-v1)
        # 2*dot(C-v1, v2-v1) = |v2-v1|^2
        # 2*dot(C-v1, v3-v1) = |v3-v1|^2
        
        v21 = v2 - v1
        v31 = v3 - v1
        v21_len2 = np.dot(v21, v21)
        v31_len2 = np.dot(v31, v31)
        v21_v31 = np.dot(v21, v31)
        
        # Determinant of the 2x2 system
        # [ dot(v21,v21)  dot(v21,v31) ] [ alpha ]   [ 0.5 * |v21|^2 ]
        # [ dot(v31,v21)  dot(v31,v31) ] [ beta  ] = [ 0.5 * |v31|^2 ]
        
        det = v21_len2 * v31_len2 - v21_v31 * v21_v31
        if np.abs(det) < 1e-12:
            return None, 0.0 # Degenerate
            
        alpha = (0.5 * v21_len2 * v31_len2 - 0.5 * v31_len2 * v21_v31) / det
        beta = (0.5 * v31_len2 * v21_len2 - 0.5 * v21_len2 * v21_v31) / det
        
        C = v1 + alpha * v21 + beta * v31
        u = C / np.linalg.norm(C)
        
        # Calculate radius
        cos_R = np.clip(np.dot(u, v1), -1.0, 1.0)
        R = np.arccos(cos_R)
        
        # Check if circumcenter is inside the spherical triangle
        # The center u is inside if it is on the same side of each edge as the opposite vertex
        # Edge 12: normal n12 = v1 x v2. v3 should be on same side as u?
        # Actually, for small triangles, we can just check if the angles are < 90
        # Or check if u dot (v2 x v3) has same sign as v1 dot (v2 x v3)
        
        n23 = np.cross(v2, v3)
        n31 = np.cross(v3, v1)
        n12 = np.cross(v1, v2)
        
        # u should be in the cone defined by v1, v2, v3
        # Simple check: u dot v1, u dot v2, u dot v3 should be close to cos_R
        # And u should be a convex combination?
        
        # If obtuse, the optimal center is the midpoint of the longest side
        # Let's check side lengths
        d12 = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        d23 = np.arccos(np.clip(np.dot(v2, v3), -1.0, 1.0))
        d31 = np.arccos(np.clip(np.dot(v3, v1), -1.0, 1.0))
        
        max_d = max(d12, d23, d31)
        
        # If max angle is > 90 deg (pi/2), the circumcenter is outside
        # For a triangle on sphere, if one angle > 90, the circumcenter is outside
        # The angle at v1 is related to the normal.
        # Approximation: if R > max_d / 2 ? No.
        
        # Let's just use the circumcenter for now, assuming triangles are small enough to be acute.
        # With N=50, average area is small.
        
        return u, R

    # Main optimization loop
    step_size = 0.05
    for iteration in range(1000):
        if time.time() > deadline:
            break
            
        try:
            hull = ConvexHull(points)
        except Exception:
            # If hull fails, jitter points
            points += np.random.normal(0, 0.01, points.shape)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            continue

        facets = hull.simplices
        radii = np.zeros(len(facets))
        centers = [None] * len(facets)
        
        for idx, facet in enumerate(facets):
            v1, v2, v3 = points[facet]
            u, R = get_circumcenter_and_radius(v1, v2, v3)
            radii[idx] = R
            centers[idx] = u

        max_R = np.max(radii)
        
        # Find facets with radius close to max
        threshold = max_R - 0.01 # radians
        worst_indices = np.where(radii > threshold)[0]
        
        if len(worst_indices) == 0:
            break
            
        # Gradient accumulation
        grad = np.zeros_like(points)
        
        for idx in worst_indices:
            facet = facets[idx]
            u = centers[idx]
            if u is None:
                continue
            v1, v2, v3 = points[facet]
            
            # Move vertices towards u
            # The direction to reduce radius is towards u
            # But we must stay on the sphere.
            # Tangent space projection:
            # d = u - v
            # tangent = d - (d dot v) v
            # step = tangent * step_size
            
            for v_idx in facet:
                v = points[v_idx]
                d = u - v
                # Project d to tangent space at v
                d_tangent = d - np.dot(d, v) * v
                grad[v_idx] += d_tangent

        # Normalize gradient and apply
        norm_grad = np.linalg.norm(grad, axis=1, keepdims=True)
        norm_grad[norm_grad == 0] = 1.0
        grad = grad / norm_grad
        
        points = points + step_size * grad
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Reduce step size if oscillating?
        # For now, keep constant or decay
        if iteration % 100 == 0:
            step_size *= 0.8

    # Final polish: try to reduce max_R further with smaller steps
    step_size = 0.01
    for iteration in range(200):
        if time.time() > deadline:
            break
        try:
            hull = ConvexHull(points)
        except Exception:
            break
            
        facets = hull.simplices
        radii = np.zeros(len(facets))
        centers = [None] * len(facets)
        
        for idx, facet in enumerate(facets):
            v1, v2, v3 = points[facet]
            u, R = get_circumcenter_and_radius(v1, v2, v3)
            radii[idx] = R
            centers[idx] = u

        max_R = np.max(radii)
        threshold = max_R - 0.005
        worst_indices = np.where(radii > threshold)[0]
        
        if len(worst_indices) == 0:
            break
            
        grad = np.zeros_like(points)
        for idx in worst_indices:
            facet = facets[idx]
            u = centers[idx]
            if u is None: continue
            for v_idx in facet:
                v = points[v_idx]
                d = u - v
                d_tangent = d - np.dot(d, v) * v
                grad[v_idx] += d_tangent

        norm_grad = np.linalg.norm(grad, axis=1, keepdims=True)
        norm_grad[norm_grad == 0] = 1.0
        grad = grad / norm_grad
        
        points = points + step_size * grad
        points = points / np.linalg.norm(points, axis=1, keepdims=True)

    return points
