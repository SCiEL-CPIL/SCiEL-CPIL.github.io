import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    # Initial points using Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = points / np.linalg.norm(points, axis=1, keepdims=True)

    def compute_covering_radius(pts):
        """Compute the actual covering radius using convex hull facets."""
        try:
            hull = ConvexHull(pts)
            max_radius_sq = 0.0
            for simplex in hull.simplices:
                facet_pts = pts[simplex]
                # Compute circumcenter of the facet (on the sphere)
                # For a triangle on the unit sphere, the circumcenter is the
                # normalized vector sum of the vertices weighted appropriately
                # Actually, for a spherical triangle, the circumcenter is the
                # intersection of the perpendicular bisectors
                # Simpler: find the point on the sphere equidistant from all 3 vertices
                # This is the normalized cross product approach won't work directly
                # Use: circumcenter direction = normalized (a + b + c) only for equilateral
                # General: solve for c such that |c - a| = |c - b| = |c - c|, |c|=1
                # This means c·a = c·b = c·c_facet
                # Let c = alpha*a + beta*b + gamma*c_facet
                # c·a = c·b => alpha + beta*(a·b) + gamma*(c·a) = beta + alpha*(a·b) + gamma*(c·b)
                # c·a = c·c_facet => alpha + beta*(a·c) + gamma*(c·a) = gamma + alpha*(c·a) + beta*(c·b)
                
                a, b, c = facet_pts
                # Set up linear system for circumcenter on unit sphere
                # The circumcenter direction satisfies:
                # c·a = c·b = c·c
                # |c| = 1
                # Let c = x*a + y*b + z*c
                # c·a = x + y*(a·b) + z*(c·a)
                # c·b = x*(a·b) + y + z*(c·b)  
                # c·c = x*(a·c) + y*(b·c) + z
                # Setting c·a = c·b:
                # x(1 - a·b) + y(a·b - 1) + z(c·a - c·b) = 0
                # Setting c·a = c·c:
                # x(1 - a·c) + y(a·c - b·c) + z(c·a - 1) = 0
                
                ab = np.dot(a, b)
                ac = np.dot(a, c)
                bc = np.dot(b, c)
                
                # Coefficient matrix
                A = np.array([
                    [1 - ab, ab - 1, ac - bc],
                    [1 - ac, ac - bc, ac - 1]
                ])
                # Solve for x, y, z (up to scale)
                # Use SVD to find null space
                U, S, Vt = np.linalg.svd(A)
                circum_dir = Vt[-1]
                circum_dir = circum_dir / np.linalg.norm(circum_dir)
                
                # Verify it's on the correct side
                if np.dot(circum_dir, a) < np.dot(circum_dir, -a):
                    circum_dir = -circum_dir
                
                # Angular distance from circumcenter to any vertex
                cos_angle = np.clip(np.dot(circum_dir, a), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                angle_sq = 2.0 * (1.0 - cos_angle)
                
                if angle_sq > max_radius_sq:
                    max_radius_sq = angle_sq
            
            return np.sqrt(max_radius_sq)
        except Exception:
            return np.pi  # fallback: full hemisphere

    def facet_circumcenters(pts):
        """Compute circumcenter directions for all facets."""
        try:
            hull = ConvexHull(pts)
            centers = []
            for simplex in hull.simplices:
                facet_pts = pts[simplex]
                a, b, c = facet_pts
                ab = np.dot(a, b)
                ac = np.dot(a, c)
                bc = np.dot(b, c)
                
                A = np.array([
                    [1 - ab, ab - 1, ac - bc],
                    [1 - ac, ac - bc, ac - 1]
                ])
                U, S, Vt = np.linalg.svd(A)
                circum_dir = Vt[-1]
                circum_dir = circum_dir / np.linalg.norm(circum_dir)
                
                if np.dot(circum_dir, a) < 0:
                    circum_dir = -circum_dir
                
                centers.append(circum_dir)
            return np.array(centers), hull.simplices
        except Exception:
            return None, None

    # Phase 1: Coarse optimization with facet-based gradient
    lr = 0.05
    best_points = points.copy()
    best_radius = compute_covering_radius(points)
    
    for it in range(2000):
        if time.time() > deadline - 10.0:
            break
        
        centers, simplices = facet_circumcenters(points)
        if centers is None:
            break
        
        # For each point, compute the average pull toward its facet circumcenters
        # A point appears in multiple facets; move it toward the circumcenter of
        # the facet with the largest circumradius
        grad = np.zeros_like(points)
        
        # Compute circumradius for each facet
        facet_radii = np.zeros(len(centers))
        for idx, center in enumerate(centers):
            a = points[simplices[idx][0]]
            cos_angle = np.clip(np.dot(center, a), -1.0, 1.0)
            facet_radii[idx] = np.arccos(cos_angle)
        
        # For each point, find the facet with max radius that contains it
        for j in range(N):
            # Find all facets containing point j
            max_r = 0
            target_center = None
            for idx in range(len(centers)):
                if j in simplices[idx]:
                    if facet_radii[idx] > max_r:
                        max_r = facet_radii[idx]
                        target_center = centers[idx]
            
            if target_center is not None:
                # Move toward the circumcenter
                # The gradient direction on the sphere is the component of
                # (center - point) tangent to the sphere
                tangent = target_center - np.dot(target_center, points[j]) * points[j]
                tangent = tangent / (np.linalg.norm(tangent) + 1e-10)
                grad[j] = tangent * facet_radii[idx]
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_norm[grad_norm < 1e-10] = 1.0
        grad = grad / grad_norm
        
        # Update
        points = points + lr * grad
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Check improvement
        current_radius = compute_covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        
        # Decay learning rate
        lr *= 0.995
        
        if it % 50 == 0:
            pass  # monitoring disabled for speed

    # Phase 2: Fine-tuning with smaller steps
    points = best_points.copy()
    lr = 0.01
    for it in range(3000):
        if time.time() > deadline - 5.0:
            break
        
        centers, simplices = facet_circumcenters(points)
        if centers is None:
            break
        
        grad = np.zeros_like(points)
        facet_radii = np.zeros(len(centers))
        for idx, center in enumerate(centers):
            a = points[simplices[idx][0]]
            cos_angle = np.clip(np.dot(center, a), -1.0, 1.0)
            facet_radii[idx] = np.arccos(cos_angle)
        
        for j in range(N):
            max_r = 0
            target_center = None
            best_idx = -1
            for idx in range(len(centers)):
                if j in simplices[idx]:
                    if facet_radii[idx] > max_r:
                        max_r = facet_radii[idx]
                        target_center = centers[idx]
                        best_idx = idx
            
            if target_center is not None:
                tangent = target_center - np.dot(target_center, points[j]) * points[j]
                norm_t = np.linalg.norm(tangent)
                if norm_t > 1e-10:
                    tangent = tangent / norm_t
                    grad[j] = tangent * facet_radii[best_idx]
        
        grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_norm[grad_norm < 1e-10] = 1.0
        grad = grad / grad_norm
        
        points = points + lr * grad
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        current_radius = compute_covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        
        lr *= 0.998

    # Final validation
    final_radius = compute_covering_radius(best_points)
    if final_radius < best_radius:
        best_radius = final_radius
    
    # Ensure points are properly normalized
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
