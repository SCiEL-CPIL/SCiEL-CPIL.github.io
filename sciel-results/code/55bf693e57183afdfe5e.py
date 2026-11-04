import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _facet_circumradii(pts, simplices):
    """
    Compute circumradii and circumcenters for all hull facets (vectorized).
    Returns:
        radii: (M,) array of angular circumradii
        centers: (M, 3) array of circumcenters on sphere
        mask: (M,) boolean array of valid facets
    """
    M = len(simplices)
    p1 = pts[simplices[:, 0]]
    p2 = pts[simplices[:, 1]]
    p3 = pts[simplices[:, 2]]
    
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

def _covering_radius_rad(pts):
    try:
        hull = ConvexHull(pts)
        radii, centers, valid = _facet_circumradii(pts, hull.simplices)
        if not np.any(valid):
            return np.pi
        return float(np.max(radii[valid]))
    except Exception:
        return np.pi

def _covering_radius_deg(pts):
    return _covering_radius_rad(pts) * 180.0 / np.pi

def _smooth_max_objective_and_grad(x_flat, temperature=0.02):
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

def _electrostatic_refine(pts, n_iters=100, lr=0.005):
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

def _lloyd_iteration(points):
    """
    Perform one iteration of Lloyd's algorithm.
    1. Compute convex hull.
    2. For each point, find the circumcenters of the triangles incident to it.
    3. Move the point to the normalized sum of these circumcenters.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        
        sums = np.zeros((N, 3))
        counts = np.zeros(N)
        
        for tri in facets:
            i, j, k = tri
            p1 = points[i]
            p2 = points[j]
            p3 = points[k]
            
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.cross(v1, v2)
            norm_cross = np.linalg.norm(cross)
            
            if norm_cross < 1e-12:
                continue
                
            c = cross / norm_cross
            
            if np.dot(c, p1) < 0:
                c = -c
            
            sums[i] += c
            sums[j] += c
            sums[k] += c
            
            counts[i] += 1
            counts[j] += 1
            counts[k] += 1
        
        new_points = points.copy()
        for i in range(N):
            if counts[i] > 0:
                new_points[i] = sums[i]
                norm = np.linalg.norm(new_points[i])
                if norm > 1e-12:
                    new_points[i] /= norm
                else:
                    new_points[i] = points[i]
            
        return new_points
    except Exception:
        return points

def _fibonacci_spiral(offset=0.0):
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _nelder_mead_polish(points, max_facets=20, max_iter=100, time_limit=5.0):
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

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Start with Fibonacci spiral
    points = _fibonacci_spiral()
    points = _normalize(points)
    
    best_points = points.copy()
    best_radius = _covering_radius_deg(points)
    
    # Phase 1: Run Lloyd's algorithm for initial convergence
    lloyd_deadline = start_time + 50.0
    stall_count = 0
    max_stall = 15
    
    while time.time() < lloyd_deadline:
        new_points = _lloyd_iteration(points)
        new_points = _normalize(new_points)
        
        current_radius = _covering_radius_deg(new_points)
        
        if current_radius < best_radius - 1e-10:
            best_radius = current_radius
            best_points = new_points.copy()
            points = new_points.copy()
            stall_count = 0
        else:
            points = new_points.copy()
            stall_count += 1
            
            if stall_count > max_stall:
                num_perturb = max(5, N // 10)
                indices = np.random.choice(N, num_perturb, replace=False)
                noise = np.random.normal(0, 0.02, (num_perturb, 3))
                points[indices] += noise
                points[indices] = _normalize(points[indices])
                stall_count = 0
    
    # Phase 2: Gradient descent with smooth surrogate
    grad_deadline = start_time + 115.0
    points = best_points.copy()
    
    current_points = points.copy()
    best_gd_radius = _covering_radius_deg(current_points)
    best_gd_points = current_points.copy()
    
    temperature = 0.04
    stall_count = 0
    max_stall = 40
    
    while time.time() < grad_deadline:
        surr_val, grad = _smooth_max_objective_and_grad(current_points.flatten(), temperature)
        
        # Simple gradient step
        step_size = 0.005
        new_flat = current_points.flatten() - step_size * grad
        new_points = _normalize(new_flat.reshape(N, 3))
        
        current_radius = _covering_radius_deg(new_points)
        
        if current_radius < best_gd_radius - 1e-12:
            best_gd_radius = current_radius
            best_gd_points = new_points.copy()
            current_points = new_points.copy()
            stall_count = 0
            temperature *= 0.995
            if temperature < 0.005:
                temperature = 0.005
        else:
            stall_count += 1
            if stall_count > max_stall:
                # Perturb worst facets
                try:
                    hull = ConvexHull(current_points)
                    facets = hull.simplices
                    worst_idx = 0
                    worst_r = 0
                    for idx, tri in enumerate(facets):
                        p1, p2, p3 = current_points[tri[0]], current_points[tri[1]], current_points[tri[2]]
                        v1, v2 = p2 - p1, p3 - p1
                        cross = np.cross(v1, v2)
                        nc = np.linalg.norm(cross)
                        if nc < 1e-12:
                            continue
                        c = cross / nc
                        if np.dot(c, p1) < 0:
                            c = -c
                        dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
                        r = np.arccos(dot_val)
                        if r > worst_r:
                            worst_r = r
                            worst_idx = idx
                    tri = facets[worst_idx]
                    for vi in tri:
                        noise = np.random.normal(0, 0.01, 3)
                        current_points[vi] += noise
                    current_points = _normalize(current_points)
                except Exception:
                    pass
                stall_count = 0
                temperature = max(temperature, 0.01)
            
            if temperature < 0.005:
                break
    
    if best_gd_radius < best_radius:
        best_radius = best_gd_radius
        best_points = best_gd_points.copy()
    
    # Phase 3: Nelder-Mead polish on worst facets
    polish_deadline = start_time + 140.0
    points = best_points.copy()
    
    while time.time() < polish_deadline:
        polished = _nelder_mead_polish(points, max_facets=15, max_iter=80, time_limit=4.0)
        
        current_radius = _covering_radius_deg(polished)
        
        if current_radius < best_radius - 1e-10:
            best_radius = current_radius
            best_points = polished.copy()
            points = polished.copy()
        else:
            # Perturb and try again
            try:
                hull = ConvexHull(points)
                radii, centers, valid = _facet_circumradii(points, hull.simplices)
                if np.any(valid):
                    r_valid = radii[valid]
                    f_valid = hull.simplices[valid]
                    max_idx = np.argmax(r_valid)
                    worst_facet = f_valid[max_idx]
                    for vi in worst_facet:
                        noise = np.random.normal(0, 0.005, 3)
                        points[vi] += noise
                    points = _normalize(points)
            except Exception:
                break
            
            if time.time() > polish_deadline - 3:
                break
    
    # Final check
    final_radius = _covering_radius_deg(best_points)
    if final_radius < best_radius:
        best_radius = final_radius
    
    return best_points
