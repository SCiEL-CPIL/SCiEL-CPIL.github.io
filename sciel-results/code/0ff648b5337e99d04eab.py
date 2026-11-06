"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

# EVOLVE-BLOCK-START

def _spherical_circumcenter(v1, v2, v3):
    """
    Compute the circumcenter of a spherical triangle on the unit sphere.
    Returns (circumcenter, circumradius) where circumcenter is a unit vector.
    """
    e1 = v2 - v1
    e2 = v3 - v1
    n = np.cross(e1, e2)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-10:
        c = (v1 + v2 + v3) / 3.0
        cn = np.linalg.norm(c)
        if cn < 1e-10:
            return np.zeros(3), 0.0
        c = c / cn
        d = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
        return c, d
    
    n = n / n_norm
    
    A = np.array([
        [e1.dot(v1) - e1.dot(v2), e2.dot(v1) - e2.dot(v2)],
        [e1.dot(v1) - e1.dot(v3), e2.dot(v1) - e2.dot(v3)]
    ])
    b = np.array([
        v1.dot(v2) - 1.0,
        v1.dot(v3) - 1.0
    ])
    
    try:
        uv = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        c = (v1 + v2 + v3) / 3.0
        cn = np.linalg.norm(c)
        if cn < 1e-10:
            return np.zeros(3), 0.0
        c = c / cn
        d = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
        return c, d
        
    C = v1 + uv[0]*e1 + uv[1]*e2
    C_norm = np.linalg.norm(C)
    if C_norm < 1e-10:
        return np.zeros(3), 0.0
    C = C / C_norm
    
    dot = np.clip(np.dot(C, v1), -1.0, 1.0)
    r = np.arccos(dot)
    
    # Ensure we get the smaller circumradius (the cap center, not the antipodal)
    # The circumcenter of a convex hull facet should be on the "outside" 
    # which corresponds to the smaller cap
    if r > np.pi / 2:
        r = np.pi - r
        C = -C
    
    return C, r


def _covering_radius(points):
    """
    Compute the covering radius in radians.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
        
    max_r = 0.0
    for f in hull.simplices:
        _, r = _spherical_circumcenter(points[f[0]], points[f[1]], points[f[2]])
        if r > max_r:
            max_r = r
            
    return max_r


def _lloyd_step(points):
    """
    One step of Lloyd's algorithm on the sphere.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return points
        
    point_facets = {i: [] for i in range(N)}
    for f in hull.simplices:
        for idx in f:
            point_facets[idx].append(f)
            
    new_points = np.zeros_like(points)
    for i in range(N):
        if not point_facets[i]:
            new_points[i] = points[i]
            continue
            
        centroid = np.zeros(3)
        count = 0
        for f in point_facets[i]:
            C, _ = _spherical_circumcenter(points[f[0]], points[f[1]], points[f[2]])
            if np.linalg.norm(C) > 1e-10:
                centroid += C
                count += 1
                
        if count > 0:
            centroid /= count
            norm = np.linalg.norm(centroid)
            if norm > 1e-10:
                new_points[i] = centroid / norm
            else:
                new_points[i] = points[i]
        else:
            new_points[i] = points[i]
            
    return new_points


def _refine_covering(points, max_iterations=300):
    """
    Refine the configuration to minimize the covering radius.
    """
    best_points = points.copy()
    best_r = _covering_radius(points)
    
    for it in range(max_iterations):
        try:
            hull = ConvexHull(points)
        except Exception:
            break
            
        max_r = 0.0
        max_facet = None
        max_cc = None
        
        for idx, f in enumerate(hull.simplices):
            C, r = _spherical_circumcenter(points[f[0]], points[f[1]], points[f[2]])
            if r > max_r:
                max_r = r
                max_facet = f
                max_cc = C
                
        if max_facet is None or max_cc is None:
            break
            
        step = 0.1 + 0.1 * np.cos(it * 0.05)
        
        new_points = points.copy()
        for idx in max_facet:
            v = new_points[idx]
            new_v = v + step * (max_cc - v)
            new_v = new_v / np.linalg.norm(new_v)
            new_points[idx] = new_v
            
        new_r = _covering_radius(new_points)
        if new_r < best_r - 1e-10:
            best_r = new_r
            best_points = new_points.copy()
            points = new_points
        else:
            if it > 50:
                break
                
    return best_points


def _targeted_facet_refine(points, max_iterations=500):
    """
    Targeted local search: find the worst facet, then perform a micro-polish
    on its three vertices to minimize that facet's circumradius.
    """
    best_points = points.copy()
    best_r = _covering_radius(points)
    no_improve_count = 0
    
    for it in range(max_iterations):
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        max_r = 0.0
        max_facet = None
        max_cc = None
        
        for f in hull.simplices:
            C, r = _spherical_circumcenter(points[f[0]], points[f[1]], points[f[2]])
            if r > max_r:
                max_r = r
                max_facet = f
                max_cc = C
        
        if max_facet is None or max_cc is None:
            break
        
        improved = False
        for trial in range(3):
            new_points = points.copy()
            for idx in max_facet:
                v = new_points[idx]
                theta = 0.05 / (trial + 1)
                d = np.arccos(np.clip(np.dot(v, max_cc), -1.0, 1.0))
                if d < 1e-10:
                    continue
                t = theta / d
                t = np.clip(t, 0.0, 1.0)
                sin_d = np.sin(d)
                new_v = (np.sin((1-t)*d) / sin_d) * v + (np.sin(t*d) / sin_d) * max_cc
                new_v = new_v / np.linalg.norm(new_v)
                new_points[idx] = new_v
            
            new_r = _covering_radius(new_points)
            if new_r < best_r - 1e-12:
                best_r = new_r
                best_points = new_points.copy()
                points = new_points
                improved = True
                no_improve_count = 0
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 20:
                break
    
    return best_points


def _smooth_objective(points, beta=20.0):
    """
    Compute a smooth approximation of the covering radius using log-sum-exp.
    We use a grid of points on the sphere and compute the soft-max of distances.
    """
    # Use a moderately dense grid for the smooth objective
    # Fibonacci sphere with ~500 points for evaluation
    M = 500
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute angular distances from each grid point to each configuration point
    # dist[i,j] = arccos(dot(grid[i], points[j]))
    dots = grid @ points.T  # shape (M, N)
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)  # shape (M, N)
    
    # For each grid point, find the minimum distance to any configuration point
    min_dists = np.min(dists, axis=1)  # shape (M,)
    
    # Smooth max via log-sum-exp
    # We want to approximate max(min_dists)
    # Use log-sum-exp on the min_dists
    max_val = np.max(min_dists)
    if max_val < 1e-10:
        return 0.0, np.zeros(3, N)
    
    # log-sum-exp trick
    shifted = min_dists - max_val
    log_sum_exp = max_val + np.log(np.sum(np.exp(beta * shifted)))
    
    # Actually, for a smooth max, we use:
    # soft_max(x) = (1/beta) * log(sum(exp(beta * x)))
    smooth_max = (1.0 / beta) * np.log(np.sum(np.exp(beta * min_dists)))
    
    return smooth_max, min_dists


def _gradient_descent_smooth(points, beta=20.0, max_iter=200, lr=0.01):
    """
    Gradient descent on a smooth surrogate of the covering radius.
    Uses finite differences for gradients.
    """
    best_points = points.copy()
    best_val = _covering_radius(points)
    
    for it in range(max_iter):
        # Compute smooth objective and gradient via finite differences
        val, _ = _smooth_objective(points, beta)
        
        if val < best_val:
            best_val = val
            best_points = points.copy()
        
        # Compute gradient using central differences
        grad = np.zeros_like(points)
        eps = 1e-4
        for i in range(N):
            for j in range(3):
                pts_plus = points.copy()
                pts_plus[i, j] += eps
                pts_plus[i] /= np.linalg.norm(pts_plus[i])
                val_plus, _ = _smooth_objective(pts_plus, beta)
                
                pts_minus = points.copy()
                pts_minus[i, j] -= eps
                pts_minus[i] /= np.linalg.norm(pts_minus[i])
                val_minus, _ = _smooth_objective(pts_minus, beta)
                
                grad[i, j] = (val_plus - val_minus) / (2 * eps)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        
        # Update with projection onto sphere
        new_points = points - lr * grad
        for i in range(N):
            new_points[i] /= np.linalg.norm(new_points[i])
        
        # Check if improvement
        new_r = _covering_radius(new_points)
        if new_r < best_val:
            points = new_points
        else:
            # Try smaller step
            for factor in [0.5, 0.25, 0.1]:
                new_points = points - lr * factor * grad
                for i in range(N):
                    new_points[i] /= np.linalg.norm(new_points[i])
                new_r = _covering_radius(new_points)
                if new_r < best_val:
                    points = new_points
                    break
            else:
                break
    
    return best_points


def _multi_start_refine(points, max_restarts=5, time_budget=None):
    """
    Multi-start refinement: try multiple random perturbations and keep the best.
    """
    start_time = time.time()
    best_points = points.copy()
    best_r = _covering_radius(points)
    
    for restart in range(max_restarts):
        if time_budget is not None and (time.time() - start_time) > time_budget:
            break
        
        # Random perturbation
        perturbed = points.copy()
        for i in range(N):
            # Small random perturbation
            noise = np.random.randn(3) * 0.05
            perturbed[i] = perturbed[i] + noise
            perturbed[i] /= np.linalg.norm(perturbed[i])
        
        # Refine
        refined = _refine_covering(perturbed, max_iterations=100)
        r = _covering_radius(refined)
        
        if r < best_r:
            best_r = r
            best_points = refined.copy()
    
    return best_points


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Initial Configuration: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Lloyd's Algorithm (Centroidal Voronoi on Sphere)
    for _ in range(50):
        if time.time() > deadline - 80:
            break
        points = _lloyd_step(points)
    
    # 3. Refinement: Reduce Covering Radius
    if time.time() < deadline - 40:
        points = _refine_covering(points, max_iterations=200)
    
    # 4. Targeted facet refinement with micro-polish
    if time.time() < deadline - 20:
        points = _targeted_facet_refine(points, max_iterations=500)
    
    # 5. Smooth gradient descent on surrogate objective
    if time.time() < deadline - 10:
        points = _gradient_descent_smooth(points, beta=20.0, max_iter=100, lr=0.005)
    
    # 6. Final targeted refinement
    if time.time() < deadline - 3:
        points = _targeted_facet_refine(points, max_iterations=200)
    
    # Final check and normalization
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    return points

# EVOLVE-BLOCK-END
