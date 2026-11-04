import numpy as np
from scipy.spatial import ConvexHull
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

def _soft_max_objective_and_grad(points, beta=50.0):
    """
    Compute the soft-max (log-sum-exp) surrogate for the max circumradius
    and its gradient with respect to point positions.
    Optimized for speed.
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
    
    # Vectorized gradient accumulation is tricky with variable indices, 
    # but N is small (130) and M is ~260, so a tight loop is acceptable 
    # if we avoid redundant calculations.
    
    for k in range(len(f_valid)):
        w_k = weights[k]
        if w_k < 1e-8: # Skip negligible weights
            continue
        
        tri = f_valid[k]
        c = c_valid[k]
        r_k = r_valid[k]
        
        if r_k < 1e-12:
            continue
        
        # Gradient of r_k w.r.t p_j for j in tri
        # d(arccos(c.p))/dp = - (c - (c.p)p) / sin(r)
        
        # We compute the gradient f_obj/dp = sum w_k * grad_r_k
        # grad_r_k = - (c - (c.p)p) / sin(r)
        
        for idx, p in zip(tri, [points[tri[0]], points[tri[1]], points[tri[2]]]):
            dot_cp = np.dot(c, p)
            sin_r = np.sqrt(max(0.0, 1.0 - dot_cp * dot_cp))
            if sin_r > 1e-12:
                # grad_p = -(c - dot_cp * p) / sin_r
                grad_p = -(c - dot_cp * p) / sin_r
                # Project to tangent plane
                grad_p -= np.dot(grad_p, p) * p
                grad[idx] += w_k * grad_p
    
    return float(log_sum_exp), grad

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin

    def time_left():
        return deadline - time.time()

    rng = np.random.default_rng(42)
    
    # Phase 0: Generate diverse initializations
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
    for _ in range(3):
        pts = rng.normal(0, 1, (N, 3))
        pts = _normalize(pts)
        inits.append(pts)
    
    # 3. Electrostatic refinement of Fibonacci
    def electrostatic_refine(pts, n_iters=100, lr=0.01):
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
    
    if time_left() > 10.0:
        elec_pts = electrostatic_refine(fib_pts, n_iters=200, lr=0.005)
        inits.append(elec_pts)
    
    # Evaluate initializations and pick the best
    best_points = None
    best_radius = np.pi
    
    for init in inits:
        if time_left() < 5.0:
            break
        rad = _covering_radius_rad(init)
        if rad < best_radius:
            best_radius = rad
            best_points = init.copy()
    
    if best_points is None:
        best_points = fib_pts
        best_radius = _covering_radius_rad(fib_pts)
    
    # Phase 1: Main optimization loop with adaptive strategy
    points = best_points.copy()
    
    # Strategy parameters
    use_gradient = False
    gradient_step = 0.02
    beta = 30.0
    lloyd_step = 0.03
    min_step = 1e-7
    no_improve_count = 0
    max_no_improve = 30
    
    while time_left() > 20.0:
        try:
            hull = ConvexHull(points)
            radii, centers, valid = _facet_circumradii(points, hull.simplices)
            if not np.any(valid):
                break
        except Exception:
            # Hull failed, kick and continue
            kick = rng.normal(0, 0.02, points.shape)
            points += kick
            points = _normalize(points)
            continue
        
        M = len(radii)
        r_valid = radii[valid]
        c_valid = centers[valid]
        f_valid = hull.simplices[valid]
        
        if use_gradient:
            # Gradient descent on soft-max
            obj, grad = _soft_max_objective_and_grad(points, beta)
            
            # Project gradient to tangent plane
            grad_tangent = np.zeros_like(points)
            for i in range(N):
                g = grad[i]
                g_tan = g - np.dot(g, points[i]) * points[i]
                norm_g = np.linalg.norm(g_tan)
                if norm_g > 1e-12:
                    grad_tangent[i] = g_tan / norm_g
            
            new_points = points - gradient_step * grad_tangent
            new_points = _normalize(new_points)
            
            current_radius = _covering_radius_rad(new_points)
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = new_points.copy()
                points = new_points
                no_improve_count = 0
                gradient_step = min(gradient_step * 1.05, 0.05)
                beta = min(beta * 1.03, 150.0)
            else:
                gradient_step = max(gradient_step * 0.85, min_step)
                no_improve_count += 1
        else:
            # Lloyd-style: move points towards their worst facet center
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
                no_improve_count = 0
                lloyd_step = min(lloyd_step * 1.04, 0.08)
            else:
                lloyd_step = max(lloyd_step * 0.82, min_step)
                no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            if not use_gradient:
                use_gradient = True
                gradient_step = 0.02
                beta = 30.0
            else:
                # Random kick to escape local minimum
                kick = rng.normal(0, 0.04, points.shape)
                points += kick
                points = _normalize(points)
                lloyd_step = 0.02
                gradient_step = 0.01
                beta = 20.0
                no_improve_count = 0
                current_radius = _covering_radius_rad(points)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_points = points.copy()
            no_improve_count = 0
    
    # Phase 2: High-precision polish
    points = best_points.copy()
    polish_deadline = deadline - 5.0
    polish_step = 0.003
    
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
        
        # Move points in the worst facet towards the circumcenter
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
            polish_step *= 0.6
            if polish_step < 1e-7:
                break
            points = best_points.copy()
    
    return best_points
