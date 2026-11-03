"""
Rewritten program for covering the sphere with 50 equal caps.
Focuses on smooth surrogate optimization (log-sum-exp) with tangent-space gradients.
"""

import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import math

N = 50

# EVOLVE-BLOCK-START

def _normalize(pts):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _covering_radius_exact(pts):
    """
    Compute the exact covering radius in radians.
    This is used for final scoring and validation, not for the smooth optimization loop.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: if hull fails, use a dense grid approximation
        return _covering_radius_grid(pts)
    
    max_cr = 0.0
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        # Compute circumcenter of the spherical triangle (a,b,c)
        # The circumcenter is the intersection of the perpendicular bisectors.
        # In 3D, it lies on the line through the origin and the cross product of (a-b) and (b-c).
        v = np.cross(a - b, b - c)
        norm_v = np.linalg.norm(v)
        
        if norm_v < 1e-12:
            # Degenerate triangle, check distances to vertices
            d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
            d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
            d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
            cr = max(d_ab, d_bc, d_ca) / 2.0
            max_cr = max(max_cr, cr)
            continue
        
        u = v / norm_v
        
        # Determine the correct direction for u (should point to the interior of the cap)
        # The normal to the face abc is n = cross(b-a, c-a)
        n_face = np.cross(b - a, c - a)
        norm_n = np.linalg.norm(n_face)
        if norm_n < 1e-12:
            continue
        n_face = n_face / norm_n
        
        # Orient normal outward from origin
        if np.dot(n_face, a) < 0:
            n_face = -n_face
            
        # The circumcenter should be on the same side as the face normal relative to the origin
        # Actually, for the covering radius, we want the center of the largest empty cap.
        # The circumcenter u might point "inwards" or "outwards". 
        # If dot(u, n_face) < 0, u points to the opposite side of the sphere from the face center.
        # We want the point on the sphere furthest from a,b,c within the Voronoi cell.
        # If the triangle is acute, the circumcenter is inside. If obtuse, it's outside.
        # The covering radius for a facet is the distance from the circumcenter to the vertices.
        # We must ensure we pick the pole that is "inside" the cap defined by the facet.
        
        if np.dot(u, n_face) > 0:
            cc = u
        else:
            cc = -u
            
        # Calculate distances
        d_a = np.arccos(np.clip(np.dot(cc, a), -1, 1))
        d_b = np.arccos(np.clip(np.dot(cc, b), -1, 1))
        d_c = np.arccos(np.clip(np.dot(cc, c), -1, 1))
        cr = max(d_a, d_b, d_c)
        
        max_cr = max(max_cr, cr)
        
    return max_cr

def _covering_radius_grid(pts, n_theta=100, n_phi=100):
    """Fallback covering radius computation using a spherical grid."""
    pts = _normalize(pts)
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2 * np.pi, n_phi)
    THETA, PHI = np.meshgrid(thetas, phis)
    x = np.sin(THETA) * np.cos(PHI)
    y = np.sin(THETA) * np.sin(PHI)
    z = np.cos(THETA)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    grid = _normalize(grid)
    
    dots = grid @ pts.T
    dots = np.clip(dots, -1, 1)
    min_dots = np.max(dots, axis=1)
    min_angles = np.arccos(min_dots)
    return np.max(min_angles)

def _fibonacci_points(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _soft_max_covering_radius_and_grad(pts, beta=50.0):
    """
    Compute the soft-max covering radius and its gradient.
    Objective: (1/beta) * log(sum(exp(beta * cr_i)))
    where cr_i is the covering radius of the i-th facet.
    """
    pts = _normalize(pts)
    n = len(pts)
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        # If hull fails, return a large penalty and zero gradient
        return 1.0, np.zeros((n, 3))
    
    simplices = hull.simplices
    if len(simplices) == 0:
        return 1.0, np.zeros((n, 3))
    
    a = pts[simplices[:, 0]]
    b = pts[simplices[:, 1]]
    c = pts[simplices[:, 2]]
    
    # Compute vectors for circumcenters
    v = np.cross(a - b, b - c)
    norm_v = np.linalg.norm(v, axis=1, keepdims=True)
    valid_v = norm_v.squeeze() > 1e-12
    norm_v_safe = np.where(valid_v, norm_v, 1.0)
    u = v / norm_v_safe
    
    # Compute face normals
    n_face = np.cross(b - a, c - a)
    norm_n = np.linalg.norm(n_face, axis=1, keepdims=True)
    valid_n = norm_n.squeeze() > 1e-12
    norm_n_safe = np.where(valid_n, norm_n, 1.0)
    n_face = n_face / norm_n_safe
    
    # Orient normals outward
    dot_na = np.sum(n_face * a, axis=1)
    flip = dot_na < 0
    n_face[flip] *= -1
    
    # Determine circumcenter direction
    dot_un = np.sum(u * n_face, axis=1)
    cc = u.copy()
    cc[dot_un <= 0] *= -1
    
    # Calculate distances to vertices for each facet
    dots_a = np.sum(cc * a, axis=1)
    dots_b = np.sum(cc * b, axis=1)
    dots_c = np.sum(cc * c, axis=1)
    
    dots_a = np.clip(dots_a, -1, 1)
    dots_b = np.clip(dots_b, -1, 1)
    dots_c = np.clip(dots_c, -1, 1)
    
    d_a = np.arccos(dots_a)
    d_b = np.arccos(dots_b)
    d_c = np.arccos(dots_c)
    crs = np.maximum(np.maximum(d_a, d_b), d_c)
    
    # Filter valid facets
    valid_all = valid_v & valid_n
    if not np.all(valid_all):
        crs = crs[valid_all]
        cc = cc[valid_all]
        a = a[valid_all]
        b = b[valid_all]
        c = c[valid_all]
        simplices = simplices[valid_all]
        
    if len(crs) == 0:
        return 1.0, np.zeros((n, 3))
        
    max_cr = np.max(crs)
    
    if max_cr < 1e-10:
        return max_cr, np.zeros((n, 3))
        
    # Soft-max calculation
    scaled = beta * (crs - max_cr)
    # Numerical stability: exp(scaled) might underflow, which is fine
    exp_scaled = np.exp(scaled)
    sum_exp = np.sum(exp_scaled)
    
    if sum_exp < 1e-300:
        # Fallback if underflow occurred
        return max_cr, np.zeros((n, 3))
        
    log_sum_exp = max_cr * beta + np.log(sum_exp)
    soft_max_cr = log_sum_exp / beta
    
    # Weights for gradient
    weights = exp_scaled / sum_exp
    
    # Compute gradient
    # d(soft_max_cr)/d(pts_i) = sum_k (w_k * d(cr_k)/d(pts_i))
    # cr_k = max(d_a_k, d_b_k, d_c_k)
    # d(d_x)/d(p) = - (cc - p (dot(cc, p))) / sin(d_x)  ... wait, gradient of arccos(dot)
    # Let f = arccos(dot(p, c)). df/dp = - (c - p dot(p,c)) / sqrt(1 - dot^2)
    # The term (c - p dot(p,c)) is the tangent component of c w.r.t p.
    
    grad = np.zeros((n, 3))
    
    for k in range(len(simplices)):
        w = weights[k]
        if w < 1e-15:
            continue
            
        cc_k = cc[k]
        simp = simplices[k]
        
        # Check which distance is the max for this facet
        # We need the gradient of the max, which is the gradient of the active constraint
        # However, since we are using a smooth approximation, we can sum gradients weighted by soft-max probs,
        # but strictly, the gradient of max is only for the argmax.
        # For the log-sum-exp surrogate, the gradient is a weighted average of the gradients of the individual crs.
        # cr_k = max(d_a, d_b, d_c). The gradient of cr_k is the gradient of the argmax distance.
        # To be differentiable everywhere, we can use the soft-argmax weights for the vertices too, 
        # or just use the gradient of the max (which is non-differentiable at ties).
        # Given the smoothness of the log-sum-exp, using the gradient of the specific d that is max is standard.
        
        # Let's compute gradients for d_a, d_b, d_c
        def grad_arccos(p, c):
            dot_val = np.dot(p, c)
            dot_val = np.clip(dot_val, -1 + 1e-15, 1 - 1e-15)
            sin_d = np.sqrt(1 - dot_val * dot_val)
            if sin_d < 1e-15:
                return np.zeros(3)
            # grad = - (c - p dot(p,c)) / sin_d
            vec = c - dot_val * p
            return -vec / sin_d
            
        # Determine which vertex is furthest for facet k
        da = d_a[k] if k < len(d_a) else 0.0
        db = d_b[k] if k < len(d_b) else 0.0
        dc = d_c[k] if k < len(d_c) else 0.0
        
        # Use soft weights for the vertices to make it smoother? 
        # For now, let's stick to the gradient of the max distance (subgradient)
        # This might cause noise, but L-BFGS-B can handle it.
        
        max_d = max(da, db, dc)
        
        # Gradient w.r.t a
        if abs(da - max_d) < 1e-8:
            g_a = grad_arccos(a[k], cc_k)
        else:
            g_a = np.zeros(3)
            
        # Gradient w.r.t b
        if abs(db - max_d) < 1e-8:
            g_b = grad_arccos(b[k], cc_k)
        else:
            g_b = np.zeros(3)
            
        # Gradient w.r.t c
        if abs(dc - max_d) < 1e-8:
            g_c = grad_arccos(c[k], cc_k)
        else:
            g_c = np.zeros(3)
            
        idx_a = simp[0]
        idx_b = simp[1]
        idx_c = simp[2]
        
        grad[idx_a] += w * g_a
        grad[idx_b] += w * g_b
        grad[idx_c] += w * g_c
        
    # Project gradient to tangent space
    # The constraint is |p|^2 = 1. The tangent space at p is { v | dot(p,v)=0 }.
    # grad_proj = grad - (dot(grad, p)) p
    grad_proj = grad - np.einsum('ij,ij->i', grad, pts)[:, None] * pts
    
    return soft_max_cr, grad_proj

def _optimize_sphere(pts, deadline):
    """
    Optimize a configuration using L-BFGS-B on the smooth surrogate.
    """
    pts = _normalize(pts)
    n = len(pts)
    x0 = pts.ravel()
    
    def objective_and_grad(x):
        p = x.reshape(n, 3)
        p = _normalize(p)
        val, grad = _soft_max_covering_radius_and_grad(p, beta=50.0)
        return val, grad.ravel()
        
    # Bounds to keep points away from origin (loose)
    bounds = [(-1.5, 1.5)] * (n * 3)
    
    result = minimize(
        objective_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={
            'maxiter': 100,
            'ftol': 1e-14,
            'gtol': 1e-10,
            'maxls': 30
        }
    )
    
    new_pts = _normalize(result.x.reshape(n, 3))
    return new_pts

def construct():
    start_time = time.time()
    # Safety margin: aim to finish within 140s to be safe against 150s limit
    deadline = start_time + 140.0
    
    best_pts = _fibonacci_points(N)
    best_cr = _covering_radius_exact(best_pts)
    
    np.random.seed(12345)
    
    # Initial diverse starts
    starts = []
    starts.append(_fibonacci_points(N))
    
    # Random start
    starts.append(_normalize(np.random.randn(N, 3)))
    
    # Jittered Fibonacci
    starts.append(_normalize(_fibonacci_points(N) + 0.2 * np.random.randn(N, 3)))
    
    # Simplex-like start (if N allows, otherwise random)
    # For N=50, we can just add more random restarts
    
    for i in range(5):
        starts.append(_normalize(np.random.randn(N, 3) * 0.5 + best_pts * 0.5))
        
    for start_pts in starts:
        if time.time() > deadline - 10:
            break
            
        # Optimize
        current_pts = _optimize_sphere(start_pts, deadline)
        
        if time.time() > deadline - 5:
            break
            
        # Evaluate exact covering radius
        cr = _covering_radius_exact(current_pts)
        
        if cr < best_cr:
            best_cr = cr
            best_pts = current_pts.copy()
            
            # If we found a good solution, do a final high-precision polish
            if time.time() < deadline - 5:
                # Re-optimize from the best solution with higher precision/iterations
                polished_pts = _optimize_sphere(best_pts, deadline)
                polished_cr = _covering_radius_exact(polished_pts)
                if polished_cr < best_cr:
                    best_cr = polished_cr
                    best_pts = polished_pts.copy()
                    
    return _normalize(best_pts)

# EVOLVE-BLOCK-END
