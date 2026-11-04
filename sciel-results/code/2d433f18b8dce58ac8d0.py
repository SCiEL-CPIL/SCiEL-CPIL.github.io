import numpy as np
from scipy.spatial import ConvexHull
import time
from scipy.optimize import minimize

N = 130

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return points / norms

def _facet_data(points):
    """Compute circumcenters and radii for all facets of the convex hull."""
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
    except Exception:
        return None, None, None
    
    p1 = points[facets[:, 0]]
    p2 = points[facets[:, 1]]
    p3 = points[facets[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = np.cross(v1, v2)
    norm_cross = np.linalg.norm(cross, axis=1)
    
    valid = norm_cross > 1e-12
    if not np.any(valid):
        return None, None, None
        
    circumcenters = np.zeros((len(facets), 3))
    radii = np.zeros(len(facets))
    
    # Only compute for valid triangles
    c = np.zeros_like(cross)
    c[valid] = cross[valid] / norm_cross[valid, np.newaxis]
    
    # Orient outward
    dots = np.einsum('ij,ij->i', c, p1)
    flip = dots < 0
    c[flip] = -c[flip]
    
    circumcenters = c
    
    dot_vals = np.einsum('ij,ij->i', c, p1)
    dot_vals = np.clip(dot_vals, -1.0, 1.0)
    radii = np.arccos(dot_vals)
    
    # Ensure radii are in [0, pi/2] (for covering radius we care about the smaller angle)
    # Actually, the angular distance on the sphere is arccos(dot), which is in [0, pi].
    # The covering radius is the max of these.
    # However, if the points are in a hemisphere, some facets might have radii > pi/2.
    # But the convex hull of points on the sphere (if not in a hemisphere) should have facets
    # whose circumcenters are "visible" from the origin in a way that the angle is < pi/2.
    # Wait, the circumcenter of a triangle on the sphere is the point on the sphere equidistant from the 3 vertices.
    # The angle is arccos(c.p).
    # If the triangle is "large", this angle can be large.
    # The covering radius is the max over all facets of this angle.
    
    return circumcenters, radii, facets

def _compute_covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    """
    try:
        circumcenters, radii, facets = _facet_data(points)
        if circumcenters is None:
            return np.pi / 2
        if len(radii) == 0:
            return 0.0
        max_radius = np.max(radii)
    except Exception:
        return np.pi / 2
        
    return np.degrees(max_radius)

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
    sum_exps = np.sum(exps)
    if sum_exps == 0:
        return smooth_val, np.zeros(3 * N)
        
    weights = exps / sum_exps
    
    # Gradient computation
    grad_flat = np.zeros(3 * N)
    
    # Precompute sin(r) for all facets
    cos_r = np.einsum('ij,ij->i', circumcenters, points[facets[:, 0]])
    sin_r = np.sqrt(np.maximum(1e-15, 1.0 - cos_r * cos_r))
    
    # grad_dir for each facet: -c / sin(r)
    grad_dirs = -circumcenters / sin_r[:, np.newaxis]
    
    # For each vertex in each facet, add weighted tangent gradient
    for idx in range(len(facets)):
        w = weights[idx]
        if w < 1e-8:
            continue
            
        g_dir = grad_dirs[idx]
        
        for v_idx in facets[idx]:
            p = points[v_idx]
            # Project g_dir to tangent space at p
            proj = np.dot(g_dir, p)
            tangent_grad = g_dir - proj * p
            grad_flat[3*v_idx:3*v_idx+3] += w * tangent_grad
            
    # Scale by 1/||x|| to account for normalization in the objective function
    grad_flat = grad_flat.reshape(N, 3) / norms
    
    return smooth_val, grad_flat.flatten()

def _electrostatic_refine(pts, n_iters=100, lr=0.01):
    """
    Electrostatic repulsion to spread points evenly.
    This helps escape local minima where points are clustered.
    """
    pts = pts.copy()
    for _ in range(n_iters):
        D = pts @ pts.T
        chord2 = 2.0 - 2.0 * D
        np.fill_diagonal(chord2, 1.0)
        W = 1.0 / chord2
        np.fill_diagonal(W, 0.0)
        row_sum = W.sum(axis=1)
        forces = pts * row_sum[:, None] - (W @ pts)
        forces = forces - np.einsum('ij,ij->i', forces, pts)[:, None] * pts
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        forces /= norms
        pts = pts + lr * forces
        pts = _normalize(pts)
    return pts

def _polish_worst_facets(points, max_iters=50, tol=1e-10):
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
        
        # Move vertices towards the circumcenter to reduce the radius
        for v_idx in tri:
            p = points[v_idx]
            # Direction to move: towards c, projected to tangent space
            direction = c - p
            proj = np.dot(direction, p)
            tangent_dir = direction - proj * p
            norm_td = np.linalg.norm(tangent_dir)
            if norm_td > 1e-12:
                tangent_dir /= norm_td
                # Small step
                points[v_idx] = points[v_idx] + 0.005 * tangent_dir
                points[v_idx] /= np.linalg.norm(points[v_idx])
                
    return points

def _nelder_mead_polish(points, max_facets=10, max_iter=200, time_limit=10.0):
    """
    Polish using Nelder-Mead on the vertices of the largest facets.
    """
    start = time.time()
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    try:
        circumcenters, radii, facets = _facet_data(points)
        if circumcenters is None:
            return best_points
        if len(radii) == 0:
            return best_points
    except Exception:
        return best_points
    
    # Get top K worst facets
    top_k = min(max_facets, len(radii))
    top_indices = np.argsort(radii)[-top_k:]
    
    vertex_indices = set()
    for idx in top_indices:
        for v in facets[idx]:
            vertex_indices.add(v)
    
    vertex_indices = list(vertex_indices)
    
    if len(vertex_indices) == 0:
        return best_points
    
    def objective(x_flat):
        if time.time() - start > time_limit:
            return np.pi
        x = x_flat.reshape(-1, 3)
        new_points = points.copy()
        for i, idx in enumerate(vertex_indices):
            new_points[idx] = x[i]
        new_points = _normalize(new_points)
        if not np.all(np.isfinite(new_points)):
            return np.pi
        return _compute_covering_radius(new_points)
    
    x0 = points[vertex_indices].flatten()
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
        if result.fun < best_radius:
            x_opt = result.x.reshape(-1, 3)
            best_points = points.copy()
            for i, idx in enumerate(vertex_indices):
                best_points[idx] = x_opt[i]
            best_points = _normalize(best_points)
            best_radius = result.fun
    except Exception:
        pass
    
    return best_points

def construct():
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _normalize(points)
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Phase 1: L-BFGS-B optimization with smooth max surrogate
    temperature = 0.05
    stall_count = 0
    max_stall = 5
    
    while time.time() < deadline - 30:
        x0 = points.flatten()
        
        def obj_and_grad(x):
            val, grad = _smooth_objective_and_grad(x, temperature)
            return val, grad
        
        try:
            result = minimize(
                obj_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 100, 'gtol': 1e-5, 'maxls': 20}
            )
            
            new_points = result.x.reshape(N, 3)
            new_points = _normalize(new_points)
            
            if not np.all(np.isfinite(new_points)):
                break
            
            new_radius = _compute_covering_radius(new_points)
            
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = new_points.copy()
                points = new_points
                temperature *= 0.8
                stall_count = 0
            else:
                temperature *= 0.7
                stall_count += 1
                if temperature < 0.001:
                    break
                    
                # Restart strategy: if stalled, perturb and try again
                if stall_count >= max_stall:
                    # Add small noise
                    noise = np.random.normal(0, 0.01, (N, 3))
                    points = points + noise
                    points = _normalize(points)
                    stall_count = 0
                    max_stall = max(3, max_stall - 1) # Reduce patience
                    
        except Exception:
            temperature *= 0.7
            if temperature < 0.001:
                break
    
    # Phase 2: Electrostatic refinement to spread points and potentially escape local minima
    if time.time() < deadline - 20:
        refined = _electrostatic_refine(best_points.copy(), n_iters=50, lr=0.01)
        refined_radius = _compute_covering_radius(refined)
        if refined_radius < best_radius:
            best_radius = refined_radius
            best_points = refined.copy()
    
    # Phase 3: Polish worst facets
    if time.time() < deadline - 10:
        polished = _polish_worst_facets(best_points.copy(), max_iters=100, tol=1e-12)
        polished_radius = _compute_covering_radius(polished)
        if polished_radius < best_radius:
            best_radius = polished_radius
            best_points = polished.copy()
    
    # Phase 4: Nelder-Mead polish on the worst facets
    if time.time() < deadline - 5:
        nm_polished = _nelder_mead_polish(best_points.copy(), max_facets=10, max_iter=100, time_limit=5.0)
        nm_radius = _compute_covering_radius(nm_polished)
        if nm_radius < best_radius:
            best_radius = nm_radius
            best_points = nm_polished.copy()
    
    # Phase 5: Final fine-tuning with very small L-BFGS-B steps
    if time.time() < deadline - 2:
        temperature = 0.001
        x0 = best_points.flatten()
        
        def obj_and_grad_final(x):
            val, grad = _smooth_objective_and_grad(x, temperature)
            return val, grad
        
        try:
            result = minimize(
                obj_and_grad_final,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 100, 'gtol': 1e-7, 'maxls': 10}
            )
            
            new_points = result.x.reshape(N, 3)
            new_points = _normalize(new_points)
            
            if np.all(np.isfinite(new_points)):
                new_radius = _compute_covering_radius(new_points)
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_points = new_points.copy()
        except Exception:
            pass
    
    return best_points
