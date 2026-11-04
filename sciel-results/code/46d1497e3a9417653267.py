import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _compute_covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        max_radius = 0.0
        
        for tri in facets:
            p1 = points[tri[0]]
            p2 = points[tri[1]]
            p3 = points[tri[2]]
            
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.cross(v1, v2)
            norm_cross = np.linalg.norm(cross)
            
            if norm_cross < 1e-12:
                continue
                
            c = cross / norm_cross
            
            if np.dot(c, p1) < 0:
                c = -c
            
            dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
            dist = np.arccos(dot_val)
            
            if dist > max_radius:
                max_radius = dist
                
    except Exception:
        return np.pi / 2
        
    return np.degrees(max_radius)

def _facet_data(points):
    """Compute circumcenters and radii for all facets of the convex hull."""
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
    except Exception:
        return None, None, None
    
    circumcenters = []
    radii = []
    
    for tri in facets:
        p1 = points[tri[0]]
        p2 = points[tri[1]]
        p3 = points[tri[2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            circumcenters.append(np.zeros(3))
            radii.append(0.0)
            continue
            
        c = cross / norm_cross
        
        if np.dot(c, p1) < 0:
            c = -c
        
        dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
        dist = np.arccos(dot_val)
        
        circumcenters.append(c)
        radii.append(dist)
    
    return np.array(circumcenters), np.array(radii), facets

def _smooth_objective_and_grad(x, temperature=0.01):
    """
    Compute smooth max of facet circumradii and its gradient.
    x is flattened array of shape (3*N,).
    """
    points = x.reshape(N, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    points = points / norms
    
    circumcenters, radii, facets = _facet_data(points)
    
    if circumcenters is None:
        return 1.5, np.zeros(3 * N)
    
    if len(radii) == 0:
        return 0.0, np.zeros(3 * N)
    
    # Smooth max
    max_r = np.max(radii)
    exps = np.exp((radii - max_r) / temperature)
    smooth_val = max_r + temperature * np.log(np.sum(exps))
    
    # Weights
    weights = exps / np.sum(exps)
    
    # Gradient computation
    grad_flat = np.zeros(3 * N)
    
    for idx, tri in enumerate(facets):
        w = weights[idx]
        if w < 1e-8:
            continue
        
        r_i = radii[idx]
        c_i = circumcenters[idx]
        
        cos_r = np.dot(c_i, points[tri[0]])
        sin_r = np.sqrt(max(1e-15, 1.0 - cos_r * cos_r))
        
        grad_dir = -c_i / sin_r
        
        for v_idx in tri:
            p = points[v_idx]
            proj = np.dot(grad_dir, p)
            tangent_grad = grad_dir - proj * p
            grad_flat[3*v_idx:3*v_idx+3] += w * tangent_grad
    
    return smooth_val, grad_flat

def _riemannian_trust_region(points, max_iters=200, tol=1e-10, initial_delta=0.05):
    """
    Riemannian trust-region optimization on the sphere.
    Uses the subgradient from the worst facet to construct a local quadratic model.
    """
    delta = initial_delta
    
    for it in range(max_iters):
        circumcenters, radii, facets = _facet_data(points)
        if circumcenters is None:
            break
        
        if len(radii) == 0:
            break
        
        max_idx = np.argmax(radii)
        max_r = radii[max_idx]
        
        if max_r < tol:
            break
        
        # Find all facets with radius close to max (active set)
        active_mask = radii > max_r - 0.01
        active_indices = np.where(active_mask)[0]
        
        # Compute subgradient: average of negative circumcenter directions
        # projected to tangent space for each vertex
        grad_tangent = np.zeros((N, 3))
        
        for idx in active_indices:
            c = circumcenters[idx]
            tri = facets[idx]
            
            # Weight by how close to max
            weight = (radii[idx] - (max_r - 0.01)) / 0.01 + 0.1
            weight = min(1.0, weight)
            
            for v_idx in tri:
                p = points[v_idx]
                # Direction toward circumcenter
                direction = c - p
                proj = np.dot(direction, p)
                tangent_dir = direction - proj * p
                norm_td = np.linalg.norm(tangent_dir)
                if norm_td > 1e-12:
                    tangent_dir /= norm_td
                    grad_tangent[v_idx] += weight * tangent_dir
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad_tangent, axis=1, keepdims=True)
        grad_norm = np.where(grad_norm < 1e-12, 1.0, grad_norm)
        grad_tangent = grad_tangent / grad_norm
        
        # Construct trust region step: move in negative gradient direction
        # with step size limited by delta
        step = -delta * grad_tangent
        
        # Project step to tangent space and normalize
        step_norms = np.linalg.norm(step, axis=1, keepdims=True)
        step_norms = np.where(step_norms < 1e-12, 1.0, step_norms)
        step = step / step_norms * delta
        
        # Apply step using exponential map on sphere
        new_points = np.zeros_like(points)
        for i in range(N):
            p = points[i]
            s = step[i]
            s_norm = np.linalg.norm(s)
            if s_norm < 1e-12:
                new_points[i] = p
            else:
                s_unit = s / s_norm
                new_points[i] = np.cos(s_norm) * p + np.sin(s_norm) * s_unit
        
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        
        # Evaluate new radius
        new_radius = _compute_covering_radius(new_points)
        old_radius = _compute_covering_radius(points)
        
        # Accept step if it improves
        if new_radius < old_radius - 1e-12:
            points = new_points
            delta = min(delta * 1.5, 0.1)
        else:
            # Reduce trust region
            delta *= 0.5
            if delta < 1e-6:
                break
    
    return points

def _polish_worst_facets(points, max_iters=100, tol=1e-12):
    """
    Iteratively polish the worst facets by moving their vertices.
    """
    for _ in range(max_iters):
        circumcenters, radii, facets = _facet_data(points)
        if circumcenters is None:
            break
        
        if len(radii) == 0:
            break
        
        max_idx = np.argmax(radii)
        max_r = radii[max_idx]
        
        if max_r < tol:
            break
        
        tri = facets[max_idx]
        c = circumcenters[max_idx]
        
        for v_idx in tri:
            p = points[v_idx]
            direction = c - p
            proj = np.dot(direction, p)
            tangent_dir = direction - proj * p
            norm_td = np.linalg.norm(tangent_dir)
            if norm_td > 1e-12:
                tangent_dir /= norm_td
                points[v_idx] = points[v_idx] + 0.01 * tangent_dir
                points[v_idx] /= np.linalg.norm(points[v_idx])
    
    return points

def construct():
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Phase 1: L-BFGS-B optimization with smooth max surrogate
    temperature = 0.05
    
    while time.time() < deadline - 40:
        x0 = points.flatten()
        
        def obj_and_grad(x):
            val, grad = _smooth_objective_and_grad(x, temperature)
            return val, grad
        
        try:
            from scipy.optimize import minimize
            result = minimize(
                obj_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 50, 'gtol': 1e-6, 'maxls': 20}
            )
            
            new_points = result.x.reshape(N, 3)
            new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
            
            if not np.all(np.isfinite(new_points)):
                break
            
            new_radius = _compute_covering_radius(new_points)
            
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = new_points.copy()
                points = new_points
                temperature *= 0.8
            else:
                temperature *= 0.7
                if temperature < 0.001:
                    break
        except Exception:
            temperature *= 0.7
            if temperature < 0.001:
                break
    
    # Phase 2: Riemannian trust-region optimization
    if time.time() < deadline - 20:
        tr_points = _riemannian_trust_region(best_points.copy(), max_iters=200, tol=1e-10, initial_delta=0.05)
        tr_radius = _compute_covering_radius(tr_points)
        if tr_radius < best_radius:
            best_radius = tr_radius
            best_points = tr_points.copy()
    
    # Phase 3: Polish worst facets
    if time.time() < deadline - 10:
        polished = _polish_worst_facets(best_points.copy(), max_iters=100, tol=1e-12)
        polished_radius = _compute_covering_radius(polished)
        if polished_radius < best_radius:
            best_radius = polished_radius
            best_points = polished.copy()
    
    # Phase 4: Final fine-tuning with very small L-BFGS-B steps
    if time.time() < deadline - 5:
        temperature = 0.001
        x0 = best_points.flatten()
        
        def obj_and_grad_final(x):
            val, grad = _smooth_objective_and_grad(x, temperature)
            return val, grad
        
        try:
            from scipy.optimize import minimize
            result = minimize(
                obj_and_grad_final,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 30, 'gtol': 1e-8, 'maxls': 10}
            )
            
            new_points = result.x.reshape(N, 3)
            new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
            
            if np.all(np.isfinite(new_points)):
                new_radius = _compute_covering_radius(new_points)
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_points = new_points.copy()
        except Exception:
            pass
    
    # Phase 5: Final high-precision Riemannian trust-region polish
    if time.time() < deadline - 3:
        final_polished = _riemannian_trust_region(best_points.copy(), max_iters=100, tol=1e-12, initial_delta=0.01)
        final_radius = _compute_covering_radius(final_polished)
        if final_radius < best_radius:
            best_radius = final_radius
            best_points = final_polished.copy()
    
    return best_points
