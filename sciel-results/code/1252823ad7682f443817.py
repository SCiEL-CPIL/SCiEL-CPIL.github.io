import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_face_data(pts):
    """
    Compute circumradius, center, and vertex indices for all faces of the convex hull.
    Returns:
    radii: array of radii
    centers: array of centers
    simplices: list of vertex indices for each face
    """
    try:
        hull = ConvexHull(pts, qhull_options='QJ')
    except Exception:
        return None, None, None
        
    simplices = hull.simplices
    n_faces = len(simplices)
    radii = np.zeros(n_faces)
    centers = np.zeros((n_faces, 3))
    
    for idx, tri in enumerate(simplices):
        a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        
        # Compute circumcenter and radius
        v1 = a - b
        v2 = b - c
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-10:
            # Degenerate case
            center = (a + b + c) / 3.0
            norm_c = np.linalg.norm(center)
            if norm_c < 1e-10:
                radii[idx] = np.pi
                centers[idx] = np.array([0,0,0])
                continue
            center = center / norm_c
            cos_r = np.clip(np.dot(center, a), -1.0, 1.0)
            radii[idx] = np.arccos(cos_r)
            centers[idx] = center
        else:
            center = cross / norm_cross
            # Ensure correct orientation (dot with centroid > 0)
            centroid = (a + b + c) / 3.0
            if np.dot(center, centroid) < 0:
                center = -center
            
            cos_r = np.clip(np.dot(center, a), -1.0, 1.0)
            radii[idx] = np.arccos(cos_r)
            centers[idx] = center
            
    return radii, centers, simplices

def _compute_covering_radius(pts):
    radii, _, _ = _compute_face_data(pts)
    if radii is None:
        return np.pi
    return np.max(radii)

def _spherical_gradient_step(pts, max_iters=10, tol=1e-8):
    """
    Perform gradient descent on the covering radius.
    The objective is R = max_k r_k.
    Subgradient is sum_{k in active} w_k * grad r_k, where w_k = 1/|active| for active faces.
    Active faces are those with r_k close to R.
    """
    best_pts = pts.copy()
    best_R = _compute_covering_radius(best_pts)
    
    for _ in range(max_iters):
        radii, centers, simplices = _compute_face_data(best_pts)
        if radii is None:
            break
            
        R = np.max(radii)
        
        # Identify active faces: those with radius > R - tol
        # Using a small tolerance to include near-active faces for smoother gradient
        active_mask = radii > (R - 1e-4)
        if not np.any(active_mask):
            break
            
        active_indices = np.where(active_mask)[0]
        n_active = len(active_indices)
        
        # Initialize gradient accumulator (tangent vectors)
        grad = np.zeros_like(pts)
        
        # Calculate gradient contribution from each active face
        for k in active_indices:
            center = centers[k]
            r_k = radii[k]
            
            # Gradient of arccos(dot(c, v)) w.r.t v is -c / sqrt(1 - dot(c,v)^2) = -c / sin(r)
            # We want to decrease r, so we move in direction -grad_r = c / sin(r)
            # Project onto tangent space at v: g = c - (c.v)v
            # The magnitude scaling is 1/sin(r)
            
            sin_r = np.sin(r_k)
            if sin_r < 1e-6:
                continue
                
            scale = 1.0 / sin_r
            
            for idx in simplices[k]:
                v = best_pts[idx]
                # Projection of center onto tangent plane at v
                dot_cv = np.dot(center, v)
                g_tangent = center - dot_cv * v
                grad[idx] += scale * g_tangent
                
        # Normalize gradient to get direction
        grad_norms = np.linalg.norm(grad, axis=1, keepdims=True)
        # Avoid division by zero
        mask_nonzero = grad_norms > 1e-12
        grad_dir = np.zeros_like(pts)
        grad_dir[mask_nonzero] = grad[mask_nonzero] / grad_norms[mask_nonzero]
        
        # Line search
        found_improvement = False
        step_size = 0.1
        for _ in range(10):
            new_pts = best_pts + step_size * grad_dir
            new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            
            new_R = _compute_covering_radius(new_pts)
            
            if new_R < best_R - 1e-10:
                best_pts = new_pts
                best_R = new_R
                found_improvement = True
                break
            else:
                step_size *= 0.5
                
        if not found_improvement:
            break
            
    return best_pts, best_R

def _local_refine(pts, deadline, max_iters=200):
    """
    Local refinement using spherical gradient descent.
    """
    best_pts = pts.copy()
    best_radius = _compute_covering_radius(best_pts)
    
    current_pts = best_pts.copy()
    
    while time.time() < deadline:
        new_pts, new_radius = _spherical_gradient_step(current_pts)
        
        if new_radius < best_radius - 1e-12:
            best_radius = new_radius
            best_pts = new_pts.copy()
            current_pts = new_pts
        else:
            # If no improvement, try a smaller perturbation or break
            # Since we did line search, no improvement usually means we are at a local minimum
            # or the gradient was zero.
            break
            
    return best_pts, best_radius

def construct():
    start_time = time.time()
    # Safety margin: 150s limit. We use 145s for computation.
    deadline = start_time + 145.0
    
    np.random.seed(42)
    
    # Initial points: Fibonacci sphere
    points = _fibonacci_sphere(N)
    best_points = points.copy()
    best_radius = _compute_covering_radius(best_points)
    
    # Phase 1: Refine the initial Fibonacci sphere
    if time.time() < deadline - 10:
        refined_pts, refined_radius = _local_refine(points.copy(), deadline - 10, max_iters=50)
        if refined_radius < best_radius:
            best_radius = refined_radius
            best_points = refined_pts.copy()
            
    # Phase 2: Multiple restarts with different initializations
    # We will try random restarts and perturbations of the best found so far
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        if restart == 0:
            # Perturb best slightly
            pts = best_points + np.random.randn(N, 3) * 0.1
        elif restart == 1:
            # Random start
            pts = np.random.randn(N, 3)
        elif restart == 2:
            # Fibonacci with phase shift
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i + np.pi / 4
            r = np.sqrt(1.0 - z * z)
            pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        elif restart == 3:
            # Another Fibonacci phase
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i + np.pi / 2
            r = np.sqrt(1.0 - z * z)
            pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        else:
            # Perturb best with increasing noise
            noise_scale = 0.05 + 0.02 * (restart % 5)
            pts = best_points + np.random.randn(N, 3) * noise_scale
            
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        
        # Refine
        if time.time() < deadline - 2:
            refined_pts, refined_radius = _local_refine(pts, deadline - 2, max_iters=50)
            if refined_radius < best_radius:
                best_radius = refined_radius
                best_points = refined_pts.copy()
                
    # Final polish on the best result
    if time.time() < deadline - 1:
        refined_pts, refined_radius = _local_refine(best_points.copy(), deadline - 1, max_iters=100)
        if refined_radius < best_radius:
            best_radius = refined_radius
            best_points = refined_pts.copy()
            
    best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
