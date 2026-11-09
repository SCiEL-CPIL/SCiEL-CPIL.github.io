import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_circumradius_and_center(a, b, c):
    """
    Compute the circumradius (in radians) and circumcenter of a spherical triangle
    defined by unit vectors a, b, c.
    """
    ab = np.clip(np.dot(a, b), -1.0, 1.0)
    bc = np.clip(np.dot(b, c), -1.0, 1.0)
    ca = np.clip(np.dot(c, a), -1.0, 1.0)
    
    v1 = a - b
    v2 = b - c
    cross = np.cross(v1, v2)
    
    if np.linalg.norm(cross) < 1e-10:
        center = a + b + c
        norm = np.linalg.norm(center)
        if norm < 1e-10:
            return np.pi, np.array([0,0,0])
        center = center / norm
        cos_r = np.clip(np.dot(center, a), -1.0, 1.0)
        return np.arccos(cos_r), center

    center = cross / np.linalg.norm(cross)
    
    centroid = (a + b + c) / 3.0
    if np.dot(center, centroid) < 0:
        center = -center
        
    cos_r = np.clip(np.dot(center, a), -1.0, 1.0)
    r = np.arccos(cos_r)
    
    return r, center

def _compute_covering_metrics(pts):
    """
    Compute the covering radius and identify the worst facets.
    """
    try:
        hull = ConvexHull(pts, qhull_options='QJ')
    except Exception:
        return np.pi, [], [], []
    
    simplices = hull.simplices
    radii = []
    centers = []
    
    for tri in simplices:
        a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        r, center = _compute_circumradius_and_center(a, b, c)
        radii.append(r)
        centers.append(center)
        
    if not radii:
        return np.pi, [], [], []
        
    max_radius = max(radii)
    threshold = max_radius - 0.001
    worst_indices = [i for i, r in enumerate(radii) if r > threshold]
    worst_facets = [simplices[i] for i in worst_indices]
    
    return max_radius, worst_facets, radii, centers

def _spherical_gradient_step(pts, step_size, beta=100.0):
    """
    Perform a Riemannian gradient descent step.
    """
    max_radius, worst_facets, radii, centers = _compute_covering_metrics(pts)
    
    if not worst_facets:
        return pts
        
    weights = np.exp(beta * (np.array(radii) - max_radius))
    
    grad = np.zeros_like(pts)
    
    for k, tri in enumerate(worst_facets):
        w = weights[k]
        if w < 1e-6:
            continue
        center = centers[k]
        for idx in tri:
            v = pts[idx]
            dot_cv = np.dot(center, v)
            g = center - dot_cv * v
            grad[idx] += w * g
            
    new_pts = pts.copy()
    for i in range(N):
        g_norm = np.linalg.norm(grad[i])
        if g_norm > 1e-12:
            g = grad[i] / g_norm
            new_pts[i] = pts[i] + step_size * g
            new_pts[i] /= np.linalg.norm(new_pts[i])
            
    return new_pts

def _local_refine(pts, deadline, max_iters=5000, initial_step=0.05, beta=50.0):
    """
    Local refinement using Riemannian gradient descent.
    """
    best_pts = pts.copy()
    best_radius, _, _, _ = _compute_covering_metrics(best_pts)
    
    step_size = initial_step
    no_improve_count = 0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        new_pts = _spherical_gradient_step(pts, step_size, beta)
        
        new_radius, _, _, _ = _compute_covering_metrics(new_pts)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = new_pts.copy()
            pts = new_pts
            no_improve_count = 0
            step_size *= 0.999
        else:
            no_improve_count += 1
            step_size *= 0.9
            if step_size < 1e-6:
                break
            if no_improve_count > 50:
                break
                
    return best_pts, best_radius

def _dual_uniformization_step(pts, step_size, beta=10.0):
    """
    Dual uniformization: try to make the face areas (solid angles) more uniform.
    """
    max_radius, worst_facets, radii, centers = _compute_covering_metrics(pts)
    
    if not worst_facets:
        return pts
        
    # For dual uniformization, we want to balance the "influence" of each face.
    # A simple heuristic: move points away from the centers of large-radius faces.
    # But a better dual approach is to optimize the face normals.
    
    # Let's use a simpler strategy: if a face has a large radius, it means the empty cap is large.
    # The center of that cap is the circumcenter.
    # We want to move the points bounding that face towards that circumcenter? No, that would shrink the face.
    # Actually, the covering radius is the max over all points y of min_i d(y, p_i).
    # The worst points y are the circumcenters of the largest faces.
    # So we should move the vertices of the worst faces *towards* their circumcenter to reduce the radius.
    
    grad = np.zeros_like(pts)
    
    for k, tri in enumerate(worst_facets):
        w = np.exp(beta * (radii[k] - max_radius))
        if w < 1e-6:
            continue
        center = centers[k]
        for idx in tri:
            v = pts[idx]
            # Move v towards center
            g = center - v
            g -= np.dot(g, v) * v  # Project to tangent space
            grad[idx] += w * g
            
    new_pts = pts.copy()
    for i in range(N):
        g_norm = np.linalg.norm(grad[i])
        if g_norm > 1e-12:
            g = grad[i] / g_norm
            new_pts[i] = pts[i] + step_size * g
            new_pts[i] /= np.linalg.norm(new_pts[i])
            
    return new_pts

def _local_refine_dual(pts, deadline, max_iters=5000, initial_step=0.05, beta=10.0):
    """
    Local refinement using dual uniformization.
    """
    best_pts = pts.copy()
    best_radius, _, _, _ = _compute_covering_metrics(best_pts)
    
    step_size = initial_step
    no_improve_count = 0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        new_pts = _dual_uniformization_step(pts, step_size, beta)
        
        new_radius, _, _, _ = _compute_covering_metrics(new_pts)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = new_pts.copy()
            pts = new_pts
            no_improve_count = 0
            step_size *= 0.999
        else:
            no_improve_count += 1
            step_size *= 0.9
            if step_size < 1e-6:
                break
            if no_improve_count > 50:
                break
                
    return best_pts, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    np.random.seed(42)
    
    # Initial points: Fibonacci sphere
    points = _fibonacci_sphere(N)
    best_points = points.copy()
    best_radius, _, _, _ = _compute_covering_metrics(best_points)
    
    # Phase 1: Refine from Fibonacci
    if time.time() < deadline - 30:
        refined_pts, refined_radius = _local_refine(points.copy(), deadline - 30, max_iters=2000, initial_step=0.1, beta=50.0)
        if refined_radius < best_radius:
            best_radius = refined_radius
            best_points = refined_pts.copy()
            
    # Phase 2: Dual uniformization
    if time.time() < deadline - 20:
        refined_pts, refined_radius = _local_refine_dual(best_points.copy(), deadline - 20, max_iters=2000, initial_step=0.05, beta=10.0)
        if refined_radius < best_radius:
            best_radius = refined_radius
            best_points = refined_pts.copy()

    # Phase 3: Multiple restarts with perturbations
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            pts = best_points + np.random.randn(N, 3) * 0.05
        elif restart == 1:
            pts = best_points + np.random.randn(N, 3) * 0.1
        elif restart == 2:
            pts = np.random.randn(N, 3)
        elif restart == 3:
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i + np.pi / 4
            r = np.sqrt(1.0 - z * z)
            pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        elif restart == 4:
            random_pts = np.random.randn(N, 3)
            random_pts /= np.linalg.norm(random_pts, axis=1, keepdims=True)
            pts = 0.8 * best_points + 0.2 * random_pts
        elif restart == 5:
            pts = best_points + np.random.randn(N, 3) * 0.15
        elif restart == 6:
            pts = np.random.randn(N, 3)
        else:
            pts = best_points + np.random.randn(N, 3) * 0.25
            
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        
        # Refine using dual uniformization
        if time.time() < deadline - 5:
            refined_pts, refined_radius = _local_refine_dual(pts, deadline - 5, max_iters=1000, initial_step=0.05, beta=10.0)
            if refined_radius < best_radius:
                best_radius = refined_radius
                best_points = refined_pts.copy()
                
    # Final polish
    if time.time() < deadline - 2:
        refined_pts, refined_radius = _local_refine_dual(best_points.copy(), deadline - 2, max_iters=1000, initial_step=0.01, beta=20.0)
        if refined_radius < best_radius:
            best_radius = refined_radius
            best_points = refined_pts.copy()
            
    best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
