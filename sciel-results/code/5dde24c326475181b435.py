import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import minimize
import math

N = 130

def _normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def _spherical_circumradius(p1, p2, p3):
    """
    Calculate the circumradius of a spherical triangle defined by 3 unit vectors.
    """
    G = np.array([
        [np.dot(p1, p1), np.dot(p1, p2), np.dot(p1, p3)],
        [np.dot(p2, p1), np.dot(p2, p2), np.dot(p2, p3)],
        [np.dot(p3, p1), np.dot(p3, p2), np.dot(p3, p3)]
    ])
    
    try:
        v = np.linalg.solve(G, np.ones(3))
    except np.linalg.LinAlgError:
        return 1.0
    
    M = np.array([p1, p2, p3])
    w = np.dot(v, M)
    
    w_norm = np.linalg.norm(w)
    if w_norm < 1e-9:
        return 1.0
        
    cos_R = 1.0 / w_norm
    if cos_R > 1.0:
        cos_R = 1.0
    R = np.arccos(cos_R)
    return R

def _covering_radius(points):
    """
    Calculate the covering radius of the configuration.
    """
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        facets = hull.simplices
        
        max_r = 0.0
        worst_facet_idx = -1
        for i in range(len(facets)):
            f = facets[i]
            p1 = points[f[0]]
            p2 = points[f[1]]
            p3 = points[f[2]]
            r = _spherical_circumradius(p1, p2, p3)
            if r > max_r:
                max_r = r
                worst_facet_idx = i
        
        return max_r, worst_facet_idx, facets[worst_facet_idx] if worst_facet_idx >= 0 else None
    except Exception:
        return 1.0, -1, None

def _soft_min_logsumexp(x, beta=50.0):
    """
    Soft minimum using log-sum-exp.
    For large beta, approximates min(x).
    """
    x_min = np.min(x)
    shifted = x - x_min
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return x_min - np.log(np.sum(np.exp(-beta * shifted))) / beta

def _soft_max_logsumexp(x, beta=50.0):
    """
    Soft maximum using log-sum-exp.
    For large beta, approximates max(x).
    """
    x_max = np.max(x)
    shifted = x - x_max
    log_sum = np.log(np.sum(np.exp(beta * shifted)))
    return x_max + np.log(np.sum(np.exp(beta * shifted))) / beta

def _tammes_soft_min(pts, beta=30.0):
    """
    Soft minimum of pairwise angular distances (Tammes problem surrogate).
    We want to maximize the minimum distance, so we minimize -soft_min.
    """
    # Compute all pairwise angular distances
    dots = np.dot(pts, pts.T)
    np.fill_diagonal(dots, 1.0)
    angles = np.arccos(np.clip(dots, -1.0, 1.0))
    # Remove diagonal
    n = len(pts)
    triu_indices = np.triu_indices(n, k=1)
    pairwise_angles = angles[triu_indices]
    
    # Soft minimum
    return -_soft_min_logsumexp(pairwise_angles, beta)

def _tammes_soft_min_grad(pts, beta=30.0):
    """
    Gradient of the Tammes soft-min objective.
    """
    n = len(pts)
    dots = np.dot(pts, pts.T)
    np.fill_diagonal(dots, 1.0)
    angles = np.arccos(np.clip(dots, -1.0, 1.0))
    triu_indices = np.triu_indices(n, k=1)
    pairwise_angles = angles[triu_indices]
    
    # Soft min: f = -log(sum(exp(-beta*x_i))) / beta + min(x)
    # df/dx_i = -exp(-beta*x_i) / sum(exp(-beta*x_j))
    x_min = np.min(pairwise_angles)
    shifted = pairwise_angles - x_min
    weights = np.exp(-beta * shifted)
    weights = weights / np.sum(weights)
    
    # Gradient with respect to angles
    grad_angles = -weights
    
    # Now we need to map back to points
    # angle_ij = arccos(p_i . p_j)
    # d(angle_ij)/d(p_i) = -p_j / sqrt(1 - (p_i.p_j)^2)
    # But we need to be careful with the chain rule
    
    # For each pair (i,j), the contribution to grad_i and grad_j
    grad = np.zeros_like(pts)
    for idx, (i, j) in enumerate(zip(triu_indices[0], triu_indices[1])):
        w = grad_angles[idx]
        dot_ij = dots[i, j]
        sin_angle = np.sqrt(np.maximum(1e-12, 1.0 - dot_ij * dot_ij))
        
        # d(angle)/d(p_i) = -p_j / sin(angle)
        # But p_i is on the sphere, so we project to tangent space
        grad_i = -w * pts[j] / sin_angle
        # Project to tangent space at p_i
        grad_i = grad_i - np.dot(grad_i, pts[i]) * pts[i]
        
        grad_j = -w * pts[i] / sin_angle
        grad_j = grad_j - np.dot(grad_j, pts[j]) * pts[j]
        
        grad[i] += grad_i
        grad[j] += grad_j
    
    return grad

def _circumcenter_of_facet(p1, p2, p3):
    """
    Find the circumcenter of a spherical triangle (point on sphere equidistant from p1, p2, p3).
    """
    G = np.array([
        [np.dot(p1, p1), np.dot(p1, p2), np.dot(p1, p3)],
        [np.dot(p2, p1), np.dot(p2, p2), np.dot(p2, p3)],
        [np.dot(p3, p1), np.dot(p3, p2), np.dot(p3, p3)]
    ])
    
    try:
        v = np.linalg.solve(G, np.ones(3))
    except np.linalg.LinAlgError:
        return (p1 + p2 + p3) / 3.0
    
    M = np.array([p1, p2, p3])
    w = np.dot(v, M)
    
    w_norm = np.linalg.norm(w)
    if w_norm < 1e-9:
        return (p1 + p2 + p3) / 3.0
    
    return w / w_norm

def _facet_circumradii(points, facets):
    """
    Compute circumradii for all facets.
    """
    radii = []
    for f in facets:
        p1 = points[f[0]]
        p2 = points[f[1]]
        p3 = points[f[2]]
        r = _spherical_circumradius(p1, p2, p3)
        radii.append(r)
    return np.array(radii)

def _smooth_covering_objective(pts, beta=20.0):
    """
    Smooth surrogate for the covering radius using soft-max of facet circumradii.
    """
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
        facets = hull.simplices
        radii = _facet_circumradii(pts, facets)
        return _soft_max_logsumexp(radii, beta)
    except Exception:
        return np.pi  # Return large value if hull fails

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Phase 1: Tammes problem - maximize minimum pairwise distance using soft-min surrogate
    # This gives a high-quality uniform initialization
    def tammes_obj(pts_flat):
        pts = pts_flat.reshape(N, 3)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        return _tammes_soft_min(pts, beta=30.0)
    
    def tammes_grad(pts_flat):
        pts = pts_flat.reshape(N, 3)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        g = _tammes_soft_min_grad(pts, beta=30.0)
        return g.flatten()
    
    try:
        res = minimize(tammes_obj, points.flatten(), jac=tammes_grad, method='L-BFGS-B',
                      options={'maxiter': 100, 'gtol': 1e-6, 'maxfun': 500})
        if res.success or res.fun < tammes_obj(points.flatten()):
            points = res.x.reshape(N, 3)
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 60:
        current_r, _, _ = _covering_radius(points)
        return points
    
    # Phase 2: Local refinement - minimize covering radius by moving facet vertices toward circumcenter
    current_r, worst_idx, worst_facet = _covering_radius(points)
    best_r = current_r
    best_points = points.copy()
    stagnation_count = 0
    
    while time.time() < deadline - 10:
        if worst_facet is None:
            break
            
        p1_idx, p2_idx, p3_idx = worst_facet
        p1 = points[p1_idx]
        p2 = points[p2_idx]
        p3 = points[p2_idx]
        
        # Compute circumcenter of the worst facet
        circumcenter = _circumcenter_of_facet(p1, p2, p3)
        
        improved = False
        # Try to move vertices toward circumcenter with various step sizes
        for step in [0.2, 0.1, 0.05, 0.02, 0.01]:
            new_p1 = _normalize(p1 + step * (circumcenter - p1))
            new_p2 = _normalize(p2 + step * (circumcenter - p2))
            new_p3 = _normalize(p3 + step * (circumcenter - p3))
            
            new_points = points.copy()
            new_points[p1_idx] = new_p1
            new_points[p2_idx] = new_p2
            new_points[p3_idx] = new_p3
            
            new_r, new_worst_idx, new_worst_facet = _covering_radius(new_points)
            
            if new_r < current_r - 1e-10:
                points = new_points
                current_r = new_r
                worst_idx = new_worst_idx
                worst_facet = new_worst_facet
                improved = True
                stagnation_count = 0
                break
        
        if not improved:
            # Try random perturbations of the worst facet vertices
            for _ in range(15):
                new_points = points.copy()
                for idx in [p1_idx, p2_idx, p3_idx]:
                    p = points[idx]
                    rand_vec = np.random.randn(3)
                    rand_vec = rand_vec - np.dot(rand_vec, p) * p
                    rand_vec = _normalize(rand_vec)
                    new_p = _normalize(p + 0.03 * rand_vec)
                    new_points[idx] = new_p
                
                new_r, new_worst_idx, new_worst_facet = _covering_radius(new_points)
                if new_r < current_r - 1e-10:
                    points = new_points
                    current_r = new_r
                    worst_idx = new_worst_idx
                    worst_facet = new_worst_facet
                    improved = True
                    break
            
            if not improved:
                stagnation_count += 1
                if stagnation_count > 30:
                    # Perturb all points slightly to escape local minimum
                    points = points + np.random.randn(N, 3) * 0.005
                    points = points / np.linalg.norm(points, axis=1, keepdims=True)
                    current_r, worst_idx, worst_facet = _covering_radius(points)
                    stagnation_count = 0
        
        if current_r < best_r:
            best_r = current_r
            best_points = points.copy()
    
    # Phase 3: Final polish using smooth epigraph formulation
    if time.time() < deadline - 5:
        try:
            def epigraph_obj(x):
                pts = x[:N*3].reshape(N, 3)
                pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
                t = x[N*3]
                r = _smooth_covering_objective(pts, beta=15.0)
                # Penalize if r > t
                penalty = np.maximum(0, r - t) ** 2
                return t + penalty
            
            x0 = np.concatenate([best_points.flatten(), [best_r]])
            res = minimize(epigraph_obj, x0, method='L-BFGS-B', options={'maxiter': 50, 'gtol': 1e-8})
            if res.fun < epigraph_obj(x0):
                new_pts = res.x[:N*3].reshape(N, 3)
                new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
                new_r, _, _ = _covering_radius(new_pts)
                if new_r < best_r:
                    best_r = new_r
                    best_points = new_pts
        except Exception:
            pass
    
    return best_points
