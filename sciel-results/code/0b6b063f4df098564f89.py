import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return points / norms

def _facet_circumradii(points, hull):
    """
    Compute circumradii and circumcenters for all hull facets.
    Returns:
        radii: (M,) array of angular circumradii
        centers: (M, 3) array of circumcenters on sphere
        facets: (M, 3) array of facet indices
    """
    facets = hull.simplices
    M = len(facets)
    
    p1 = points[facets[:, 0]]
    p2 = points[facets[:, 1]]
    p3 = points[facets[:, 2]]
    
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
    
    return radii, centers, facets

def _covering_radius_rad(points):
    """Compute covering radius in radians."""
    try:
        hull = ConvexHull(points)
        radii, centers, facets = _facet_circumradii(points, hull)
        return float(np.max(radii))
    except Exception:
        return np.pi

def _covering_radius_deg(points):
    """Compute covering radius in degrees."""
    return _covering_radius_rad(points) * 180.0 / np.pi

def _fibonacci_spiral():
    """Generate Fibonacci spiral points on sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(points)

def _electrostatic_refine(points, n_iter=200, learning_rate=0.01):
    """
    Refine points using electrostatic repulsion (Riesz s-energy minimization).
    This spreads points more evenly before the covering radius optimization.
    """
    points = points.copy()
    lr = learning_rate
    
    for it in range(n_iter):
        # Compute pairwise distances and forces
        # Force between points i and j: -grad of 1/|pi-pj|
        # = -(pi - pj) / |pi - pj|^3 (for repulsion)
        
        # Vectorized computation
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero
        mask = dist > 1e-10
        dist_safe = np.where(mask, dist, 1.0)
        
        # Force: sum over j != i of (pj - pi) / |pi - pj|^3
        # This is the negative gradient of sum 1/|pi-pj|
        force = np.zeros_like(points)
        for i in range(N):
            for j in range(N):
                if i != j:
                    d = dist[i, j]
                    if d > 1e-10:
                        force[i] += (points[j] - points[i]) / (d ** 3)
        
        # Project forces to tangent space
        for i in range(N):
            f = force[i]
            f_tan = f - np.dot(f, points[i]) * points[i]
            points[i] += lr * f_tan
        
        points = _normalize(points)
        
        # Decay learning rate
        lr *= 0.99
    
    return points

def _smooth_surrogate(points, beta=50.0):
    """
    Compute soft-max (log-sum-exp) surrogate for max circumradius.
    Returns objective value and gradient.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return 1.0, np.zeros_like(points)
    
    radii, centers, facets = _facet_circumradii(points, hull)
    M = len(radii)
    
    if M == 0:
        return 0.0, np.zeros_like(points)
    
    r_max = np.max(radii)
    if r_max < 1e-15:
        return 0.0, np.zeros_like(points)
    
    shifted = beta * (radii - r_max)
    # Numerically stable log-sum-exp
    log_sum_exp = r_max + np.log(np.sum(np.exp(shifted))) / beta
    
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    
    grad = np.zeros_like(points)
    
    for k in range(M):
        if weights[k] < 1e-10:
            continue
        
        tri = facets[k]
        p1 = points[tri[0]]
        p2 = points[tri[1]]
        p3 = points[tri[2]]
        c = centers[k]
        r_k = radii[k]
        
        if r_k < 1e-12:
            continue
        
        # Gradient of arccos(c·p) w.r.t. p is -c / sqrt(1-(c·p)^2)
        for idx, p in [(tri[0], p1), (tri[1], p2), (tri[2], p3)]:
            dot_cp = np.dot(c, p)
            sin_r = np.sqrt(max(0.0, 1.0 - dot_cp * dot_cp))
            if sin_r > 1e-12:
                grad_p = -c / sin_r
                # Project to tangent space
                grad_p -= np.dot(grad_p, p) * p
                grad[idx] += weights[k] * grad_p
    
    return float(log_sum_exp), grad

def _surrogate_gradient(points, beta=50.0):
    """Get just the gradient from the smooth surrogate."""
    _, grad = _smooth_surrogate(points, beta)
    return grad

def _lm_polish(points, max_iter=100, tol=1e-10):
    """
    Levenberg-Marquardt style polishing: move vertices of worst facets
    towards circumcenters with adaptive step size.
    """
    points = points.copy()
    best_radius = _covering_radius_rad(points)
    best_points = points.copy()
    
    step_size = 0.01
    
    for it in range(max_iter):
        try:
            hull = ConvexHull(points)
            radii, centers, facets = _facet_circumradii(points, hull)
        except Exception:
            break
        
        max_idx = np.argmax(radii)
        max_r = radii[max_idx]
        
        if max_r < tol:
            break
        
        worst_facet = facets[max_idx]
        worst_center = centers[max_idx]
        
        # Try moving the worst facet vertices
        new_points = points.copy()
        for idx in worst_facet:
            v = worst_center - points[idx]
            dot_vp = np.dot(v, points[idx])
            tangent_v = v - dot_vp * points[idx]
            norm_tv = np.linalg.norm(tangent_v)
            if norm_tv > 1e-12:
                new_points[idx] += step_size * tangent_v / norm_tv
        
        new_points = _normalize(new_points)
        new_radius = _covering_radius_rad(new_points)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_points = new_points.copy()
            points = new_points
            step_size = min(step_size * 1.1, 0.05)
        else:
            step_size *= 0.7
            if step_size < 1e-12:
                break
    
    return best_points, best_radius

def _nm_polish_top_facets(points, n_facets=5, max_iter=200):
    """
    Nelder-Mead style polishing on the top-n worst facets.
    For each top facet, try small perturbations of its 3 vertices.
    """
    points = points.copy()
    best_radius = _covering_radius_rad(points)
    best_points = points.copy()
    
    for it in range(max_iter):
        try:
            hull = ConvexHull(points)
            radii, centers, facets = _facet_circumradii(points, hull)
        except Exception:
            break
        
        # Get top n facets by radius
        top_idx = np.argsort(radii)[-n_facets:]
        
        improved = False
        for k in top_idx:
            r_k = radii[k]
            if r_k < 1e-12:
                continue
            
            tri = facets[k]
            c = centers[k]
            
            # Try moving each vertex slightly toward circumcenter
            for idx in tri:
                v = c - points[idx]
                dot_vp = np.dot(v, points[idx])
                tangent_v = v - dot_vp * points[idx]
                norm_tv = np.linalg.norm(tangent_v)
                if norm_tv < 1e-12:
                    continue
                
                # Try a few step sizes
                for step in [0.005, 0.002, 0.001, 0.0005]:
                    new_points = points.copy()
                    new_points[idx] += step * tangent_v / norm_tv
                    new_points = _normalize(new_points)
                    new_radius = _covering_radius_rad(new_points)
                    
                    if new_radius < best_radius:
                        best_radius = new_radius
                        best_points = new_points.copy()
                        points = new_points
                        improved = True
                        break
        
        if not improved:
            # Try random kicks on worst facet vertices
            rng = np.random.default_rng(it)
            for k in top_idx[:2]:
                tri = facets[k]
                for idx in tri:
                    kick = rng.normal(0, 0.001, 3)
                    new_points = points.copy()
                    new_points[idx] += kick
                    new_points = _normalize(new_points)
                    new_radius = _covering_radius_rad(new_points)
                    if new_radius < best_radius:
                        best_radius = new_radius
                        best_points = new_points.copy()
                        points = new_points
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Phase 1: Initialize with Fibonacci spiral
    points = _fibonacci_spiral()
    
    # Add small random perturbation to break symmetry
    rng = np.random.default_rng(42)
    points += rng.normal(0, 0.01, points.shape)
    points = _normalize(points)
    
    # Phase 2: Electrostatic refinement to spread points
    if time.time() < deadline - 60:
        points = _electrostatic_refine(points, n_iter=100, learning_rate=0.005)
    
    best_points = points.copy()
    best_radius = _covering_radius_rad(points)
    
    # Phase 3: Greedy Lloyd-style optimization
    step_size = 0.03
    min_step_size = 1e-8
    no_improve_count = 0
    max_no_improve = 30
    
    while time.time() < deadline - 40:
        try:
            hull = ConvexHull(points)
            radii, centers, facets = _facet_circumradii(points, hull)
        except Exception:
            kick = rng.normal(0, 0.02, points.shape)
            points += kick
            points = _normalize(points)
            continue
        
        M = len(facets)
        if M == 0:
            break
        
        # For each point, find the adjacent triangle with the maximum radius
        point_max_radius = np.zeros(N)
        point_max_center = np.zeros((N, 3))
        
        for k in range(M):
            r_k = radii[k]
            if r_k <= 0:
                continue
            c_k = centers[k]
            tri = facets[k]
            for idx in tri:
                if r_k > point_max_radius[idx]:
                    point_max_radius[idx] = r_k
                    point_max_center[idx] = c_k
        
        # Lloyd-style: move points towards their worst facet center
        max_move = np.zeros_like(points)
        
        for i in range(N):
            if point_max_radius[i] > 0:
                v = point_max_center[i] - points[i]
                dot_vp = np.dot(v, points[i])
                tangent_v = v - dot_vp * points[i]
                norm_tv = np.linalg.norm(tangent_v)
                if norm_tv > 1e-12:
                    max_move[i] = (tangent_v / norm_tv) * point_max_radius[i]
        
        new_points = points + step_size * max_move
        new_points = _normalize(new_points)
        
        current_radius = _covering_radius_rad(new_points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = new_points.copy()
            points = new_points
            no_improve_count = 0
            step_size = min(step_size * 1.04, 0.08)
        else:
            step_size = max(step_size * 0.82, min_step_size)
            no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            # Random kick
            kick = rng.normal(0, 0.03, points.shape)
            points += kick
            points = _normalize(points)
            step_size = 0.02
            no_improve_count = 0
            current_radius = _covering_radius_rad(points)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
    
    # Phase 4: Smooth surrogate gradient descent
    if time.time() < deadline - 25:
        points = best_points.copy()
        grad_step = 0.01
        beta = 30.0
        no_improve_grad = 0
        
        while time.time() < deadline - 25:
            obj, grad = _smooth_surrogate(points, beta)
            
            grad_tangent = np.zeros_like(points)
            for i in range(N):
                g = grad[i]
                g_tan = g - np.dot(g, points[i]) * points[i]
                norm_g = np.linalg.norm(g_tan)
                if norm_g > 1e-12:
                    grad_tangent[i] = g_tan / norm_g
            
            new_points = points - grad_step * grad_tangent
            new_points = _normalize(new_points)
            
            current_radius = _covering_radius_rad(new_points)
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = new_points.copy()
                points = new_points
                no_improve_grad = 0
                grad_step = min(grad_step * 1.03, 0.03)
                beta = min(beta * 1.02, 100.0)
            else:
                grad_step = max(grad_step * 0.85, 1e-6)
                no_improve_grad += 1
            
            if no_improve_grad > 20:
                break
    
    # Phase 5: LM polishing
    if time.time() < deadline - 15:
        points, radius = _lm_polish(best_points, max_iter=50, tol=1e-10)
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
    
    # Phase 6: NM polishing on top facets
    if time.time() < deadline - 8:
        points, radius = _nm_polish_top_facets(best_points, n_facets=5, max_iter=100)
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
    
    # Phase 7: Final high-precision polish
    if time.time() < deadline - 3:
        points = best_points.copy()
        polish_step = 0.001
        
        while time.time() < deadline - 3:
            try:
                hull = ConvexHull(points)
                radii, centers, facets = _facet_circumradii(points, hull)
            except Exception:
                break
            
            max_idx = np.argmax(radii)
            max_r = radii[max_idx]
            worst_facet = facets[max_idx]
            worst_center = centers[max_idx]
            
            improved = False
            for idx in worst_facet:
                v = worst_center - points[idx]
                dot_vp = np.dot(v, points[idx])
                tangent_v = v - dot_vp * points[idx]
                norm_tv = np.linalg.norm(tangent_v)
                if norm_tv > 1e-12:
                    for step in [polish_step, polish_step * 0.5, polish_step * 0.1]:
                        new_points = points.copy()
                        new_points[idx] += step * tangent_v / norm_tv
                        new_points = _normalize(new_points)
                        new_radius = _covering_radius_rad(new_points)
                        if new_radius < best_radius:
                            best_radius = new_radius
                            best_points = new_points.copy()
                            points = new_points
                            improved = True
                            break
            
            if not improved:
                polish_step *= 0.5
                if polish_step < 1e-8:
                    break
    
    return best_points
