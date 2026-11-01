"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    # 1. Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z]).flatten()
    
    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        return pts / norms

    def get_covering_radius_and_grads(x_flat):
        """
        Computes the covering radius and the gradient of the radius w.r.t the flattened coordinates.
        The covering radius R is the maximum over all facets F of the convex hull of the distance 
        from the circumcenter of F to the vertices of F (if the circumcenter is inside the spherical triangle).
        If the circumcenter is outside, the max distance is at the midpoint of the longest edge.
        """
        pts = x_flat.reshape(N, 3)
        pts = normalize(pts)
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            return 1.0, np.zeros_like(x_flat)
        
        max_r = 0.0
        active_facets = []
        
        # Precompute distances for all facets
        facet_data = []
        for simplex in hull.simplices:
            i, j, k = simplex
            p1, p2, p3 = pts[i], pts[j], pts[k]
            
            # Compute circumcenter of the planar triangle p1, p2, p3
            u = p2 - p1
            v = p3 - p1
            uu = np.dot(u, u)
            vv = np.dot(v, v)
            uv = np.dot(u, v)
            
            det = uu * vv - uv * uv
            if abs(det) < 1e-10:
                continue
                
            a = (0.5 * uu * vv - 0.5 * uv * vv) / det
            b = (0.5 * vv * uu - 0.5 * uu * uv) / det
            
            c_planar = p1 + a * u + b * v
            c_norm = np.linalg.norm(c_planar)
            if c_norm < 1e-10:
                continue
                
            c_sph = c_planar / c_norm
            
            # Check if c_sph is inside the spherical triangle.
            P = np.column_stack([p1, p2, p3])
            try:
                coeffs = np.linalg.solve(P, c_sph)
                if np.all(coeffs > -1e-6): # Allow small numerical error
                    is_inside = True
                else:
                    is_inside = False
            except np.linalg.LinAlgError:
                is_inside = False
                
            if is_inside:
                d_c = np.arccos(np.clip(np.dot(c_sph, p1), -1.0, 1.0))
                facet_data.append((simplex, c_sph, d_c, 'center'))
            else:
                # If outside, the max distance is at the midpoint of the longest edge
                m12 = (p1 + p2) / np.linalg.norm(p1 + p2)
                m23 = (p2 + p3) / np.linalg.norm(p2 + p3)
                m31 = (p3 + p1) / np.linalg.norm(p3 + p1)
                
                d_m12 = np.arccos(np.clip(np.dot(m12, p1), -1.0, 1.0))
                d_m23 = np.arccos(np.clip(np.dot(m23, p2), -1.0, 1.0))
                d_m31 = np.arccos(np.clip(np.dot(m31, p3), -1.0, 1.0))
                
                max_d = max(d_m12, d_m23, d_m31)
                if d_m12 == max_d:
                    facet_data.append((simplex, m12, d_m12, 'edge12'))
                elif d_m23 == max_d:
                    facet_data.append((simplex, m23, d_m23, 'edge23'))
                else:
                    facet_data.append((simplex, m31, d_m31, 'edge31'))
            
            if facet_data[-1][2] > max_r:
                max_r = facet_data[-1][2]
        
        if not facet_data:
            return 1.0, np.zeros_like(x_flat)
        
        # Identify active facets (those contributing to the max radius)
        tol = 1e-4
        for idx, (simplex, y, d, type) in enumerate(facet_data):
            if d >= max_r - tol:
                active_facets.append(idx)
        
        # Compute gradient
        grads = np.zeros_like(x_flat)
        
        for idx in active_facets:
            simplex, y, d, type = facet_data[idx]
            
            if type == 'center':
                for idx_p in simplex:
                    p = pts[idx_p]
                    dot = np.dot(y, p)
                    denom = np.sqrt(max(1e-10, 1.0 - dot*dot))
                    grads[idx_p*3 : idx_p*3+3] += -y / denom
            else:
                if type == 'edge12':
                    indices = [simplex[0], simplex[1]]
                elif type == 'edge23':
                    indices = [simplex[1], simplex[2]]
                else:
                    indices = [simplex[2], simplex[0]]
                
                for idx_p in indices:
                    p = pts[idx_p]
                    dot = np.dot(y, p)
                    denom = np.sqrt(max(1e-10, 1.0 - dot*dot))
                    grads[idx_p*3 : idx_p*3+3] += -y / denom
        
        if len(active_facets) > 0:
            grads /= len(active_facets)
            
        return max_r, grads

    def objective_with_grad(x_flat):
        val, grad = get_covering_radius_and_grads(x_flat)
        pts = x_flat.reshape(N, 3)
        pts = normalize(pts)
        
        grad_reshaped = grad.reshape(N, 3)
        dots = np.sum(grad_reshaped * pts, axis=1, keepdims=True)
        grad_tangent = grad_reshaped - dots * pts
        
        grad_tangent_flat = grad_tangent.flatten()
        
        return val, grad_tangent_flat

    # Optimization
    best_x = x0
    best_val, _ = get_covering_radius_and_grads(best_x)
    
    # Run L-BFGS-B
    try:
        res = minimize(
            objective_with_grad, 
            x0, 
            method='L-BFGS-B', 
            jac=True, 
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if time.time() < deadline:
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
    except Exception:
        pass

    # Final Polish with simple gradient descent on the sphere
    pts = best_x.reshape(N, 3)
    pts = normalize(pts)
    
    step_size = 0.01
    for it in range(100):
        if time.time() > deadline:
            break
            
        val, grad = get_covering_radius_and_grads(pts.flatten())
        
        grad_reshaped = grad.reshape(N, 3)
        dots = np.sum(grad_reshaped * pts, axis=1, keepdims=True)
        grad_tangent = grad_reshaped - dots * pts
        
        new_pts = pts - step_size * grad_tangent
        new_pts = normalize(new_pts)
        
        new_val, _ = get_covering_radius_and_grads(new_pts.flatten())
        
        if new_val < val:
            pts = new_pts
            val = new_val
            step_size *= 0.95
        else:
            step_size *= 0.8
            if step_size < 1e-5:
                break

    pts = normalize(pts)
    return pts
# EVOLVE-BLOCK-END
