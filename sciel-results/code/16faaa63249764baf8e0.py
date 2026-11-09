import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _normalize(pts):
    """Normalize vectors to unit length."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _compute_facet_data(pts):
    """
    Compute convex hull and data for each facet.
    Returns:
    - max_rad: the maximum covering radius among all facets
    - worst_simplex: indices of the vertices of the worst facet
    - facet_data: list of tuples (simplex_indices, local_rad, is_acute, circumcenter, edge_midpoints, edge_lengths)
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi, None, []
    
    max_rad = 0.0
    worst_simplex = None
    facet_data = []
    
    for simplex in hull.simplices:
        v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        # Spherical circumcenter
        normal = np.cross(v2 - v1, v3 - v1)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        c_s = normal / norm
        
        # Distances from circumcenter to vertices
        d1 = np.arccos(np.clip(np.dot(c_s, v1), -1.0, 1.0))
        d2 = np.arccos(np.clip(np.dot(c_s, v2), -1.0, 1.0))
        d3 = np.arccos(np.clip(np.dot(c_s, v3), -1.0, 1.0))
        
        # Edge midpoints and distances
        e12 = v1 + v2
        e12_norm = np.linalg.norm(e12)
        if e12_norm > 1e-10:
            m12 = e12 / e12_norm
            d_e12 = np.arccos(np.clip(np.dot(m12, v1), -1.0, 1.0))
        else:
            d_e12 = 0.0
            m12 = np.zeros(3)
            
        e23 = v2 + v3
        e23_norm = np.linalg.norm(e23)
        if e23_norm > 1e-10:
            m23 = e23 / e23_norm
            d_e23 = np.arccos(np.clip(np.dot(m23, v2), -1.0, 1.0))
        else:
            d_e23 = 0.0
            m23 = np.zeros(3)

        e31 = v3 + v1
        e31_norm = np.linalg.norm(e31)
        if e31_norm > 1e-10:
            m31 = e31 / e31_norm
            d_e31 = np.arccos(np.clip(np.dot(m31, v3), -1.0, 1.0))
        else:
            d_e31 = 0.0
            m31 = np.zeros(3)
        
        # Determine local radius and type
        # If acute (all d < pi/2), circumcenter is inside, radius is max(d1,d2,d3)
        # If obtuse, circumcenter is outside, radius is max(d_e12, d_e23, d_e31)
        # Note: For a spherical triangle, if it's obtuse, the circumcenter is outside the triangle.
        # The worst point is the midpoint of the longest edge.
        
        is_acute = max(d1, d2, d3) <= np.pi / 2.0
        
        if is_acute:
            local_rad = max(d1, d2, d3)
        else:
            local_rad = max(d_e12, d_e23, d_e31)
            
        if local_rad > max_rad:
            max_rad = local_rad
            worst_simplex = simplex
            # Store data for the worst facet for gradient calculation
            # We need to know which edge is the longest if obtuse
            # Or just store all and let gradient function decide
            facet_data.append((simplex, local_rad, is_acute, c_s, (m12, m23, m31), (d_e12, d_e23, d_e31), (d1, d2, d3)))
        else:
            facet_data.append((simplex, local_rad, is_acute, c_s, (m12, m23, m31), (d_e12, d_e23, d_e31), (d1, d2, d3)))
            
    return max_rad, worst_simplex, facet_data

def _energy(pts):
    """
    Energy is the covering radius.
    """
    rad, _, _ = _compute_facet_data(pts)
    return rad

def _energy_grad(pts):
    """
    Gradient of the covering radius.
    We only compute the gradient for the worst facet.
    """
    rad, worst_simplex, facet_data = _compute_facet_data(pts)
    
    if worst_simplex is None:
        return np.zeros_like(pts)
    
    grad = np.zeros_like(pts)
    
    # Find the data for the worst facet
    worst_data = None
    for data in facet_data:
        if np.array_equal(data[0], worst_simplex):
            worst_data = data
            break
            
    if worst_data is None:
        return np.zeros_like(pts)
        
    simplex, local_rad, is_acute, c_s, mids, d_edges, d_verts = worst_data
    v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
    
    # Case 1: Acute triangle
    # Radius R = distance from circumcenter C to vertex v1 (or v2, v3)
    # R = arccos(C . v1)
    # dR/dv1 = -C / sqrt(1 - (C . v1)^2)
    # Project to tangent space: grad_tan = grad - (grad . v1) v1
    # Note: C is fixed (depends on v1,v2,v3 but for small steps we can treat it as constant? 
    # No, C depends on v1,v2,v3.
    # However, for the "worst point" gradient, we usually assume the worst point is fixed locally.
    # If the worst point is the circumcenter, its position depends on all 3 vertices.
    # This is complex.
    # Alternative: Use the gradient of the distance to the nearest vertex for the worst point.
    # If acute, worst point is C.
    # dist(C, v1) = arccos(C . v1).
    # Gradient w.r.t v1: -C / sin(d1)
    # Gradient w.r.t v2: 0 (if C is fixed)
    # But C moves.
    
    # Let's use a simpler, robust gradient approximation often used in sphere packing:
    # If acute:
    # The force on v1 is in the direction of C - (C . v1) v1 ? No.
    # We want to decrease d(C, v1).
    # Move v1 towards C? No, C is on the sphere.
    # If we move v1 towards C, d decreases.
    # Direction: C - (C . v1) v1 is the tangent component of C at v1.
    # Wait, C is the center. We want to increase the cap.
    # Actually, we want to minimize R.
    # R = arccos(C . v1).
    # dR/dv1 = -C / sqrt(1 - (C . v1)^2) = -C / sin(R)
    # This vector points from v1 towards C (tangent component).
    # So grad = -C / sin(R) for v1?
    # Let's check signs.
    # If we move v1 in direction -grad, we move towards C.
    # Moving v1 towards C reduces the angle?
    # Yes, if v1 and C are close, moving v1 towards C makes them closer, angle decreases.
    # So grad should point AWAY from C?
    # Let's re-derive.
    # f(v) = arccos(C . v).
    # df/dv = -1/sqrt(1 - (C.v)^2) * C = -C / sin(theta).
    # So gradient points in direction -C (tangent).
    # To minimize f, we move in direction -grad = C / sin(theta).
    # So we move v1 towards C. This makes sense.
    
    if is_acute:
        # Worst point is C.
        # Gradient for v1, v2, v3
        # Note: C depends on v1, v2, v3.
        # Approximation: Treat C as constant for the gradient step?
        # This is the "worst point fixed" assumption.
        # This is valid if the facet remains the worst and the type (acute) remains the same.
        
        # Grad for v1
        sin_d1 = np.sin(d_verts[0])
        if sin_d1 > 1e-6:
            grad_v1 = -c_s / sin_d1
        else:
            grad_v1 = np.zeros(3)
            
        # Grad for v2
        sin_d2 = np.sin(d_verts[1])
        if sin_d2 > 1e-6:
            grad_v2 = -c_s / sin_d2
        else:
            grad_v2 = np.zeros(3)
            
        # Grad for v3
        sin_d3 = np.sin(d_verts[2])
        if sin_d3 > 1e-6:
            grad_v3 = -c_s / sin_d3
        else:
            grad_v3 = np.zeros(3)
            
        grad[simplex[0]] += grad_v1
        grad[simplex[1]] += grad_v2
        grad[simplex[2]] += grad_v3
        
    else:
        # Obtuse triangle.
        # Worst point is midpoint of longest edge.
        # Find longest edge
        # d_e12, d_e23, d_e31 are distances from midpoint to vertex (half edge length)
        # Longest edge corresponds to max d_e
        
        max_d_edge = max(d_edges)
        
        if np.isclose(max_d_edge, d_edges[0]):
            # Edge 12 is longest
            m = mids[0]
            # R = arccos(m . v1)
            # Grad v1 = -m / sin(d_e12)
            # Grad v2 = -m / sin(d_e12) ?
            # Wait, m = (v1+v2)/|v1+v2|.
            # This is more complex.
            # Let's use the property that for the midpoint of an edge, the gradient is symmetric.
            # Actually, if we move v1 and v2 towards m, the edge shortens.
            # Direction for v1: m - (m . v1) v1
            # Direction for v2: m - (m . v2) v2
            # These are tangent vectors pointing towards m.
            # Magnitude?
            # R = arccos(m . v1).
            # dR/dv1 = -m / sin(d_e12).
            # So grad_v1 = -m / sin(d_e12).
            # grad_v2 = -m / sin(d_e23)? No, d_e23 is for edge 23.
            # For edge 12, the distance is d_e12.
            # So grad_v1 = -m / sin(d_e12)
            # grad_v2 = -m / sin(d_e12) -- wait, d(m, v2) is also d_e12.
            
            sin_de = np.sin(d_edges[0])
            if sin_de > 1e-6:
                grad_v1 = -m / sin_de
                grad_v2 = -m / sin_de
            else:
                grad_v1 = np.zeros(3)
                grad_v2 = np.zeros(3)
                
            grad[simplex[0]] += grad_v1
            grad[simplex[1]] += grad_v2
            
        elif np.isclose(max_d_edge, d_edges[1]):
            # Edge 23 is longest
            m = mids[1]
            sin_de = np.sin(d_edges[1])
            if sin_de > 1e-6:
                grad_v2 = -m / sin_de
                grad_v3 = -m / sin_de
            else:
                grad_v2 = np.zeros(3)
                grad_v3 = np.zeros(3)
                
            grad[simplex[1]] += grad_v2
            grad[simplex[2]] += grad_v3
            
        else:
            # Edge 31 is longest
            m = mids[2]
            sin_de = np.sin(d_edges[2])
            if sin_de > 1e-6:
                grad_v3 = -m / sin_de
                grad_v1 = -m / sin_de
            else:
                grad_v3 = np.zeros(3)
                grad_v1 = np.zeros(3)
                
            grad[simplex[2]] += grad_v3
            grad[simplex[0]] += grad_v1
            
    # Project gradient to tangent space for all points
    grad_dot_p = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tangent = grad - grad_dot_p * pts
    
    return grad_tangent

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin for 150s limit
    
    best_points = None
    best_radius = np.inf
    
    np.random.seed(42)
    
    # Initial guess: Fibonacci sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points_init = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Run multiple restarts
    # Since gradient is noisy (depends on hull), we might need more restarts or careful step size.
    # L-BFGS-B should handle it if we provide a good gradient.
    
    for restart in range(30):
        if time.time() - start_time > time_limit - 10.0:
            break
            
        if restart == 0:
            x0 = points_init.copy()
        else:
            x0 = points_init.copy()
            noise = np.random.normal(0, 0.3, x0.shape)
            x0 = x0 + noise
            x0 = _normalize(x0)
            
        try:
            res = minimize(
                _energy, 
                x0, 
                method='L-BFGS-B', 
                jac=_energy_grad,
                options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 20}
            )
            opt_pts = _normalize(res.x)
            
            # Evaluate true covering radius
            rad, _, _ = _compute_facet_data(opt_pts)
            
            if rad < best_radius:
                best_radius = rad
                best_points = opt_pts.copy()
                
        except Exception as e:
            pass # Skip failed restarts

    # If no optimization succeeded, return initial
    if best_points is None:
        return points_init
        
    return best_points

# EVOLVE-BLOCK-END
