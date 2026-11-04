import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130
_DEADLINE = None

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

def _soft_max_objective_and_grad(points, beta=50.0):
    """
    Compute the soft-max (log-sum-exp) surrogate for the max circumradius
    and its gradient with respect to point positions (tangent to sphere).
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return 1.0, np.zeros_like(points)
    
    radii, centers, valid = _facet_circumradii(points, hull.simplices)
    M = len(radii)
    
    if M == 0 or not np.any(valid):
        return 0.0, np.zeros_like(points)
    
    # Use only valid facets
    r_valid = radii[valid]
    c_valid = centers[valid]
    f_valid = hull.simplices[valid]
    
    if len(r_valid) == 0:
        return 0.0, np.zeros_like(points)
    
    r_max = np.max(r_valid)
    if r_max < 1e-15:
        return 0.0, np.zeros_like(points)
    
    shifted = beta * (r_valid - r_max)
    log_sum_exp = r_max + np.log(np.sum(np.exp(shifted))) / beta
    
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    
    grad = np.zeros_like(points)
    
    # Gradient of soft-max of circumradii w.r.t. each point
    # d(r_k)/d(p_i) for point i in facet k:
    # r_k = arccos(c_k . p_i)
    # d(r_k)/d(p_i) = -c_k / sin(r_k)  (in 3D, projected to tangent)
    # But we need to be careful: the circumcenter depends on all 3 vertices
    # For a proper gradient, we need d(c_k)/d(p_j) for each vertex j
    
    # Simplified: use the fact that moving p_i toward c_k decreases r_k
    # The tangent direction that most decreases r_k is -(c_k - (c_k.p_i)p_i)/|...|
    for k in range(len(f_valid)):
        w_k = weights[k]
        if w_k < 1e-10:
            continue
        
        tri = f_valid[k]
        c = c_valid[k]
        r_k = r_valid[k]
        
        if r_k < 1e-12:
            continue
        
        for idx in tri:
            p = points[idx]
            dot_cp = np.dot(c, p)
            sin_r = np.sqrt(max(0.0, 1.0 - dot_cp * dot_cp))
            if sin_r > 1e-12:
                # Gradient of r_k w.r.t. p_i (tangent component)
                # d(r_k)/d(p_i) = -c_k / sin(r_k) but projected to tangent plane
                # The tangent component of -c/sin_r is: -(c - (c.p)p)/sin_r
                grad_p = -(c - dot_cp * p) / sin_r
                grad_p -= np.dot(grad_p, p) * p
                grad[idx] += w_k * grad_p
    
    return float(log_sum_exp), grad

def _lbfgs_polish(points, max_facets=40, max_iter=300, time_limit=8.0, beta=50.0):
    """
    Polish using L-BFGS-B on the vertices of the largest facets with soft-max objective.
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
    
    vertex_indices = sorted(list(vertex_indices))
    
    if len(vertex_indices) == 0:
        return best_points
    
    n_vars = len(vertex_indices) * 3
    
    def objective_and_grad(x_flat):
        if time.time() - start > time_limit:
            return np.pi, np.zeros(n_vars)
        x = x_flat.reshape(-1, 3)
        new_points = points.copy()
        for i, idx in enumerate(vertex_indices):
            new_points[idx] = x[i]
        new_points = _normalize(new_points)
        if not np.all(np.isfinite(new_points)):
            return np.pi, np.zeros(n_vars)
        obj, grad_full = _soft_max_objective_and_grad(new_points, beta)
        # Extract gradient for only the variable vertices
        grad_sub = np.zeros(n_vars)
        for i, idx in enumerate(vertex_indices):
            grad_sub[i*3:(i+1)*3] = grad_full[idx]
        return float(obj), grad_sub
    
    x0 = points[vertex_indices].flatten()
    
    try:
        result = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True,
                         options={'maxiter': max_iter, 'gtol': 1e-10, 'ftol': 1e-12,
                                  'maxfun': 500, 'maxls': 20})
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
    global _DEADLINE
    start_time = time.time()
    _DEADLINE = start_time + 130.0  # 20s safety margin

    def time_left():
        return _DEADLINE - time.time()

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
    
    # 3. Electrostatic refinement (faster version)
    def electrostatic_refine(pts, n_iters=80, lr=0.008):
        pts = pts.copy()
        for it in range(n_iters):
            if time_left() < 20.0:
                break
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
    
    if time_left() > 20.0:
        elec_pts = electrostatic_refine(fib_pts, n_iters=80, lr=0.008)
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
    
    # Main optimization: L-BFGS-B on all points using soft-max objective
    points = best_points.copy()
    
    def full_objective_and_grad(x_flat):
        if time_left() < 20.0:
            return np.pi, np.zeros(3 * N)
        x = x_flat.reshape(-1, 3)
        new_points = _normalize(x)
        if not np.all(np.isfinite(new_points)):
            return np.pi, np.zeros(3 * N)
        obj, grad = _soft_max_objective_and_grad(new_points, beta=40.0)
        return float(obj), grad.flatten()
    
    x0 = points.flatten()
    
    try:
        result = minimize(full_objective_and_grad, x0, method='L-BFGS-B', jac=True,
                         options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-10,
                                  'maxfun': 400, 'maxls': 30})
        if result.fun < best_radius:
            x_opt = result.x.reshape(-1, 3)
            best_points = _normalize(x_opt)
            best_radius = result.fun
    except Exception:
        pass
    
    # Local refinement: Lloyd-type moves on worst facets
    points = best_points.copy()
    lloyd_step = 0.03
    no_improve = 0
    max_no_improve = 15
    
    while time_left() > 25.0 and no_improve < max_no_improve:
        try:
            hull = ConvexHull(points)
            radii, centers, valid = _facet_circumradii(points, hull.simplices)
            if not np.any(valid):
                break
        except Exception:
            kick = rng.normal(0, 0.02, points.shape)
            points += kick
            points = _normalize(points)
            continue
        
        r_valid = radii[valid]
        c_valid = centers[valid]
        f_valid = hull.simplices[valid]
        
        # Move each point toward the circumcenter of its worst facet
        point_max_radius = np.zeros(N)
        point_max_center = np.zeros((N, 3))
        
        for k in range(len(f_valid)):
            r_k = r_valid[k]
            if r_k <= 0:
                continue
            c_k = c_valid[k]
            tri = f_valid[k]
            for idx in tri:
                if r_k > point_max_radius[idx]:
                    point_max_radius[idx] = r_k
                    point_max_center[idx] = c_k
        
        max_move = np.zeros_like(points)
        for i in range(N):
            if point_max_radius[i] > 0:
                v = point_max_center[i] - points[i]
                dot_vp = np.dot(v, points[i])
                tangent_v = v - dot_vp * points[i]
                norm_tv = np.linalg.norm(tangent_v)
                if norm_tv > 1e-12:
                    max_move[i] = (tangent_v / norm_tv) * point_max_radius[i]
        
        new_points = points + lloyd_step * max_move
        new_points = _normalize(new_points)
        
        current_radius = _covering_radius_rad(new_points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = new_points.copy()
            points = new_points
            no_improve = 0
            lloyd_step = min(lloyd_step * 1.05, 0.08)
        else:
            lloyd_step = max(lloyd_step * 0.8, 1e-6)
            no_improve += 1
    
    # Final polish with L-BFGS-B on worst facet vertices
    if time_left() > 15.0:
        polished = _lbfgs_polish(best_points, max_facets=40, max_iter=200, time_limit=10.0, beta=60.0)
        pol_radius = _covering_radius_rad(polished)
        if pol_radius < best_radius:
            best_radius = pol_radius
            best_points = polished
    
    # High-precision polish on worst facets
    points = best_points.copy()
    polish_step = 0.002
    
    while time_left() > 5.0:
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
        
        max_idx = np.argmax(r_valid)
        max_r = r_valid[max_idx]
        worst_facet = f_valid[max_idx]
        worst_center = c_valid[max_idx]
        
        for idx in worst_facet:
            v = worst_center - points[idx]
            dot_vp = np.dot(v, points[idx])
            tangent_v = v - dot_vp * points[idx]
            norm_tv = np.linalg.norm(tangent_v)
            if norm_tv > 1e-12:
                points[idx] += polish_step * tangent_v / norm_tv
        
        points = _normalize(points)
        
        current_radius = _covering_radius_rad(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        else:
            polish_step *= 0.5
            if polish_step < 1e-8:
                break
            points = best_points.copy()
    
    return best_points
