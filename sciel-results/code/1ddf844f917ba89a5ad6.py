"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _compute_covering_radius(points):
    """Compute the covering radius (max circumradius of hull facets) in radians."""
    try:
        hull = ConvexHull(points)
        max_ang = 0.0
        for simplex in hull.simplices:
            v1 = points[simplex[0]]
            v2 = points[simplex[1]]
            v3 = points[simplex[2]]
            
            edge1 = v2 - v1
            edge2 = v3 - v1
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal = normal / norm
            
            # Ensure normal points outward (away from origin)
            if np.dot(normal, v1) < 0:
                normal = -normal
                
            d1 = np.dot(normal, v1)
            d2 = np.dot(normal, v2)
            d3 = np.dot(normal, v3)
            
            # Clamp for numerical stability
            d1 = np.clip(d1, -1.0, 1.0)
            d2 = np.clip(d2, -1.0, 1.0)
            d3 = np.clip(d3, -1.0, 1.0)
            
            ang = max(np.arccos(d1), np.arccos(d2), np.arccos(d3))
            if ang > max_ang:
                max_ang = ang
                
        return max_ang
    except Exception:
        return np.pi  # Worst case

def _smooth_surrogate(points, beta=50.0):
    """
    Compute a smooth surrogate for the covering radius using log-sum-exp.
    This gives a differentiable approximation to the max circumradius.
    """
    try:
        hull = ConvexHull(points)
        angles = []
        for simplex in hull.simplices:
            v1 = points[simplex[0]]
            v2 = points[simplex[1]]
            v3 = points[simplex[2]]
            
            edge1 = v2 - v1
            edge2 = v3 - v1
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal = normal / norm
            if np.dot(normal, v1) < 0:
                normal = -normal
                
            d1 = np.clip(np.dot(normal, v1), -1.0, 1.0)
            d2 = np.clip(np.dot(normal, v2), -1.0, 1.0)
            d3 = np.clip(np.dot(normal, v3), -1.0, 1.0)
            
            ang = max(np.arccos(d1), np.arccos(d2), np.arccos(d3))
            angles.append(ang)
            
        if not angles:
            return np.pi, None
            
        angles = np.array(angles)
        # Log-sum-exp for smooth max
        max_ang = np.max(angles)
        if len(angles) > 1:
            exp_angles = np.exp(beta * (angles - max_ang))
            log_sum = max_ang + np.log(np.sum(exp_angles)) / beta
        else:
            log_sum = max_ang
            
        return log_sum, angles
    except Exception:
        return np.pi, None

def _surrogate_gradient(points, beta=50.0):
    """
    Compute gradient of the smooth surrogate with respect to points.
    Uses finite differences on the smooth surrogate.
    """
    grad = np.zeros_like(points)
    eps = 1e-4
    
    current_val, _ = _smooth_surrogate(points, beta)
    
    for i in range(N):
        for j in range(3):
            # Central difference
            pts_plus = points.copy()
            pts_plus[i, j] += eps
            pts_plus = _normalize(pts_plus)
            val_plus, _ = _smooth_surrogate(pts_plus, beta)
            
            pts_minus = points.copy()
            pts_minus[i, j] -= eps
            pts_minus = _normalize(pts_minus)
            val_minus, _ = _smooth_surrogate(pts_minus, beta)
            
            grad[i, j] = (val_plus - val_minus) / (2 * eps)
            
    return grad

def _gradient_descent(pts, max_iters=50, lr=0.01, beta=50.0, deadline=None):
    """
    Perform gradient descent on the smooth surrogate to refine the configuration.
    """
    start_time = time.time()
    current_val, _ = _smooth_surrogate(pts, beta)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        grad = _surrogate_gradient(pts, beta)
        
        # Project gradient to tangent space (remove radial component)
        for i in range(N):
            radial = np.dot(grad[i], pts[i])
            grad[i] -= radial * pts[i]
            
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        grad = grad / grad_norm
        
        # Update points
        new_pts = pts - lr * grad
        new_pts = _normalize(new_pts)
        
        new_val, _ = _smooth_surrogate(new_pts, beta)
        
        if new_val < current_val:
            pts = new_pts
            current_val = new_val
        else:
            # Reduce learning rate
            lr *= 0.8
            if lr < 1e-6:
                break
                
    return pts, current_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_radius = _compute_covering_radius(pts)
    
    # Phase 1: Gradient descent on smooth surrogate
    if time.time() < deadline - 30:
        pts, surrogate_val = _gradient_descent(pts, max_iters=80, lr=0.02, beta=50.0, deadline=deadline - 30)
        current_radius = _compute_covering_radius(pts)
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = pts.copy()
    
    # Phase 2: Greedy facet refinement
    current_radius = _compute_covering_radius(pts)
    step_size = 0.05
    
    while time.time() < deadline:
        # Find the worst facet
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        worst_facet_idx = -1
        max_ang = -1.0
        worst_normal = None
        
        for idx, simplex in enumerate(hull.simplices):
            v1 = pts[simplex[0]]
            v2 = pts[simplex[1]]
            v3 = pts[simplex[2]]
            
            edge1 = v2 - v1
            edge2 = v3 - v1
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal = normal / norm
            if np.dot(normal, v1) < 0:
                normal = -normal
            
            d1 = np.clip(np.dot(normal, v1), -1.0, 1.0)
            d2 = np.clip(np.dot(normal, v2), -1.0, 1.0)
            d3 = np.clip(np.dot(normal, v3), -1.0, 1.0)
            
            ang = max(np.arccos(d1), np.arccos(d2), np.arccos(d3))
            
            if ang > max_ang:
                max_ang = ang
                worst_facet_idx = idx
                worst_normal = normal
        
        if worst_facet_idx == -1:
            break
            
        # Get the vertices of the worst facet
        simplex = hull.simplices[worst_facet_idx]
        v1 = pts[simplex[0]]
        v2 = pts[simplex[1]]
        v3 = pts[simplex[2]]
        
        # Move these points towards the worst_normal (the center of the hole)
        alpha = step_size
        
        # Update points
        pts[simplex[0]] = v1 + alpha * worst_normal
        pts[simplex[1]] = v2 + alpha * worst_normal
        pts[simplex[2]] = v3 + alpha * worst_normal
        
        # Normalize
        pts = _normalize(pts)
        
        # Check new radius
        new_radius = _compute_covering_radius(pts)
        
        if new_radius < current_radius:
            current_radius = new_radius
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = pts.copy()
            # Decay step size slowly
            step_size *= 0.99
        else:
            # Revert
            pts[simplex[0]] = v1
            pts[simplex[1]] = v2
            pts[simplex[2]] = v3
            pts = _normalize(pts)
            
            # Reduce step size
            step_size *= 0.9
            if step_size < 1e-5:
                break
                
    # Phase 3: Final polish with smaller steps
    if time.time() < deadline - 5:
        step_size = 0.01
        for _ in range(100):
            if time.time() > deadline - 5:
                break
                
            try:
                hull = ConvexHull(pts)
            except Exception:
                break
                
            worst_facet_idx = -1
            max_ang = -1.0
            worst_normal = None
            
            for idx, simplex in enumerate(hull.simplices):
                v1 = pts[simplex[0]]
                v2 = pts[simplex[1]]
                v3 = pts[simplex[2]]
                
                edge1 = v2 - v1
                edge2 = v3 - v1
                normal = np.cross(edge1, edge2)
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                normal = normal / norm
                if np.dot(normal, v1) < 0:
                    normal = -normal
                    
                d1 = np.clip(np.dot(normal, v1), -1.0, 1.0)
                d2 = np.clip(np.dot(normal, v2), -1.0, 1.0)
                d3 = np.clip(np.dot(normal, v3), -1.0, 1.0)
                
                ang = max(np.arccos(d1), np.arccos(d2), np.arccos(d3))
                
                if ang > max_ang:
                    max_ang = ang
                    worst_facet_idx = idx
                    worst_normal = normal
            
            if worst_facet_idx == -1:
                break
                
            simplex = hull.simplices[worst_facet_idx]
            v1 = pts[simplex[0]]
            v2 = pts[simplex[1]]
            v3 = pts[simplex[2]]
            
            pts[simplex[0]] = v1 + step_size * worst_normal
            pts[simplex[1]] = v2 + step_size * worst_normal
            pts[simplex[2]] = v3 + step_size * worst_normal
            pts = _normalize(pts)
            
            new_radius = _compute_covering_radius(pts)
            if new_radius < current_radius:
                current_radius = new_radius
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = pts.copy()
            else:
                pts[simplex[0]] = v1
                pts[simplex[1]] = v2
                pts[simplex[2]] = v3
                pts = _normalize(pts)
                step_size *= 0.9
                if step_size < 1e-6:
                    break
                    
    return best_pts
# EVOLVE-BLOCK-END
