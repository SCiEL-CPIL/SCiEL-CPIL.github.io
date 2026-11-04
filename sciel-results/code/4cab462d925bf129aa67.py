import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return points / norms

def _facet_circumradii(points, simplices):
    """
    Compute circumradii and circumcenters for all hull facets (vectorized).
    Returns:
        radii: (M,) array of angular circumradii
        centers: (M, 3) array of circumcenters on sphere
        mask: (M,) boolean array of valid facets
    """
    M = len(simplices)
    p1 = points[simplices[:, 0]]
    p2 = points[simplices[:, 1]]
    p3 = points[simplices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = np.cross(v1, v2)
    norm_cross = np.linalg.norm(cross, axis=1)
    
    valid = norm_cross > 1e-14
    centers = np.zeros((M, 3))
    radii = np.zeros(M)
    
    if np.any(valid):
        c = cross[valid] / norm_cross[valid, np.newaxis]
        
        centroid = (p1[valid] + p2[valid] + p3[valid]) / 3.0
        dot_c_centroid = np.sum(c * centroid, axis=1)
        flip = dot_c_centroid < 0
        c[flip] = -c[flip]
        
        centers[valid] = c
        dot_val = np.clip(np.sum(c * p1[valid], axis=1), -1.0, 1.0)
        dist = np.arccos(dot_val)
        radii[valid] = np.where(dist > np.pi / 2, np.pi - dist, dist)
    
    return radii, centers, valid

def _covering_radius_rad(points):
    """
    Compute the covering radius of a set of points on the unit sphere (in radians).
    """
    try:
        hull = ConvexHull(points)
        radii, centers, valid = _facet_circumradii(points, hull.simplices)
        if not np.any(valid):
            return np.pi
        return float(np.max(radii[valid]))
    except Exception:
        return np.pi

def _covering_radius_deg(points):
    return _covering_radius_rad(points) * 180.0 / np.pi

def _smooth_max_objective_and_grad(x_flat, temperature=0.01):
    """
    Compute smooth max of facet circumradii and its gradient.
    x_flat is a flattened array of shape (3*N,).
    """
    points = x_flat.reshape(N, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return 1.5, np.zeros(3 * N)
    
    radii, centers, valid = _facet_circumradii(points, hull.simplices)
    
    if not np.any(valid):
        return 1.5, np.zeros(3 * N)
    
    r_valid = radii[valid]
    c_valid = centers[valid]
    f_valid = hull.simplices[valid]
    
    if len(r_valid) == 0:
        return 0.0, np.zeros(3 * N)
    
    max_r = np.max(r_valid)
    if max_r < 1e-15:
        return 0.0, np.zeros(3 * N)
    
    # Smooth max (log-sum-exp)
    shifted = (r_valid - max_r) / temperature
    exps = np.exp(shifted)
    smooth_val = max_r + temperature * np.log(np.sum(exps))
    
    weights = exps / np.sum(exps)
    
    grad_flat = np.zeros(3 * N)
    
    for k in range(len(f_valid)):
        w_k = weights[k]
        if w_k < 1e-8:
            continue
        
        tri = f_valid[k]
        c = c_valid[k]
        r_k = r_valid[k]
        
        if r_k < 1e-12:
            continue
        
        cos_r = np.dot(c, points[tri[0]])
        sin_r = np.sqrt(max(1e-15, 1.0 - cos_r * cos_r))
        
        # Gradient of arccos(c·p) w.r.t. p is -c/sin_r
        # Project to tangent space
        for idx in tri:
            p = points[idx]
            grad_dir = -c / sin_r
            proj = np.dot(grad_dir, p)
            tangent_grad = grad_dir - proj * p
            grad_flat[3*idx:3*idx+3] += w_k * tangent_grad
    
    return float(smooth_val), grad_flat

def _electrostatic_refine(pts, n_iters=150, lr=0.005):
    """Electrostatic repulsion to spread points evenly."""
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

def _nelder_mead_polish(points, max_facets=30, max_iter=200, time_limit=5.0):
    """
    Polish using Nelder-Mead on the vertices of the largest facets.
    """
    start = time.time()
    best_points = points.copy()
    best_radius = _covering_radius_rad(points)
    
    try:
        hull = ConvexHull(points)
        radii, centers, valid = _facet_circumradii(points, hull.simplices)
        if not np.any(valid):
            return best_points
    except Exception:
        return best_points
    
    r_valid = radii[valid]
    c_valid = centers[valid]
    f_valid = hull.simplices[valid]
    
    if len(r_valid) == 0:
        return best_points
    
    top_k = min(max_facets, len(r_valid))
    top_indices = np.argsort(r_valid)[-top_k:]
    
    vertex_indices = set()
    for idx in top_indices:
        for v in f_valid[idx]:
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
        return _covering_radius_rad(new_points)
    
    x0 = points[vertex_indices].flatten()
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12})
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

def _polish_worst_facets(points, max_iters=200, tol=1e-10):
    """
    Iteratively polish the worst facets by moving their vertices toward circumcenters.
    """
    for _ in range(max_iters):
        try:
            hull = ConvexHull(points)
            radii, centers, valid = _facet_circumradii(points, hull.simplices)
            if not np.any(valid):
                break
        except Exception:
            break
        
        r_valid = radii[valid]
        c_valid = centers[valid]
        f_valid = hull.simplices[valid]
        
        if len(r_valid) == 0:
            break
        
        max_idx = np.argmax(r_valid)
        max_r = r_valid[max_idx]
        
        if max_r < tol:
            break
        
        worst_facet = f_valid[max_idx]
        worst_center = c_valid[max_idx]
        
        for idx in worst_facet:
            v = worst_center - points[idx]
            dot_vp = np.dot(v, points[idx])
            tangent_v = v - dot_vp * points[idx]
            norm_tv = np.linalg.norm(tangent_v)
            if norm_tv > 1e-12:
                points[idx] += 0.01 * tangent_v / norm_tv
        
        points = _normalize(points)
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit

    def time_left():
        return deadline - time.time()

    rng = np.random.default_rng(42)
    
    # Generate initializations
    inits = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    fib_pts = _normalize(fib_pts)
    
    # Perturbed Fibonacci
    fib_pert = fib_pts + rng.normal(0, 0.05, fib_pts.shape)
    fib_pert = _normalize(fib_pert)
    
    inits.append(fib_pts)
    inits.append(fib_pert)
    
    # 2. Random points
    for _ in range(2):
        pts = rng.normal(0, 1, (N, 3))
        pts = _normalize(pts)
        inits.append(pts)
    
    # 3. Electrostatic refinement
    if time_left() > 20.0:
        elec_pts = _electrostatic_refine(fib_pts, n_iters=100, lr=0.005)
        inits.append(elec_pts)
    
    # Evaluate initializations and pick the best
    best_points = None
    best_radius = np.pi
    
    for init in inits:
        if time_left() < 15.0:
            break
        rad = _covering_radius_rad(init)
        if rad < best_radius:
            best_radius = rad
            best_points = init.copy()
    
    if best_points is None:
        best_points = fib_pts
        best_radius = _covering_radius_rad(fib_pts)
    
    # Phase 1: L-BFGS-B optimization with smooth max surrogate
    points = best_points.copy()
    temperature = 0.05
    
    while time_left() > 40.0:
        x0 = points.flatten()
        
        def obj_and_grad(x):
            val, grad = _smooth_max_objective_and_grad(x, temperature)
            return val, grad
        
        try:
            result = minimize(
                obj_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 150, 'gtol': 1e-8, 'maxls': 50, 'ftol': 1e-10}
            )
            
            new_points = result.x.reshape(N, 3)
            new_points = _normalize(new_points)
            
            if not np.all(np.isfinite(new_points)):
                break
            
            new_radius = _covering_radius_rad(new_points)
            
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = new_points.copy()
                points = new_points
                temperature *= 0.85
            else:
                temperature *= 0.7
                if temperature < 0.001:
                    break
        except Exception:
            temperature *= 0.7
            if temperature < 0.001:
                break
    
    # Phase 2: Polish worst facets
    if time_left() > 15.0:
        polished = _polish_worst_facets(best_points.copy(), max_iters=300, tol=1e-12)
        polished_radius = _covering_radius_rad(polished)
        if polished_radius < best_radius:
            best_radius = polished_radius
            best_points = polished
    
    # Phase 3: Nelder-Mead polish on top facets
    if time_left() > 10.0:
        polished = _nelder_mead_polish(best_points, max_facets=30, max_iter=300, time_limit=10.0)
        pol_radius = _covering_radius_rad(polished)
        if pol_radius < best_radius:
            best_radius = pol_radius
            best_points = polished
    
    # Phase 4: Final fine-tuning with very small L-BFGS-B steps
    if time_left() > 5.0:
        temperature = 0.001
        x0 = best_points.flatten()
        
        def obj_and_grad_final(x):
            val, grad = _smooth_max_objective_and_grad(x, temperature)
            return val, grad
        
        try:
            result = minimize(
                obj_and_grad_final,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 100, 'gtol': 1e-10, 'maxls': 40, 'ftol': 1e-12}
            )
            
            new_points = result.x.reshape(N, 3)
            new_points = _normalize(new_points)
            
            if np.all(np.isfinite(new_points)):
                new_radius = _covering_radius_rad(new_points)
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_points = new_points.copy()
        except Exception:
            pass
    
    return best_points
