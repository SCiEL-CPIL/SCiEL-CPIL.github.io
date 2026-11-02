"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize

def _normalize(pts):
    """Normalize rows of pts to unit length."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _compute_hull_data(points):
    """Compute convex hull facets and their circumcenters/radii (vectorized).
    Returns (facets, circumcenters, radii) or (None, None, None) on failure."""
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
    except Exception:
        try:
            delaunay = Delaunay(points)
            facets = delaunay.simplices
        except Exception:
            return None, None, None
    
    n_facets = len(facets)
    if n_facets == 0:
        return None, None, None
    
    a = points[facets[:, 0]]
    b = points[facets[:, 1]]
    c = points[facets[:, 2]]
    
    v1 = a - b
    v2 = b - c
    n_vec = np.cross(v1, v2)
    norm_n = np.linalg.norm(n_vec, axis=1)
    
    valid = norm_n > 1e-12
    circumcenters = np.zeros((n_facets, 3))
    radii = np.zeros(n_facets)
    
    if np.any(valid):
        cc = n_vec[valid] / norm_n[valid, np.newaxis]
        # Ensure circumcenter faces outward (toward 'a')
        dots = np.sum(cc * a[valid], axis=1)
        cc[dots < 0] *= -1
        circumcenters[valid] = cc
        
        cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
        radii[valid] = np.arccos(cos_r)
    
    return facets, circumcenters, radii

def _compute_covering_radius(points):
    """Compute the covering radius in degrees using convex hull facets."""
    facets, circumcenters, radii = _compute_hull_data(points)
    if radii is None:
        return 90.0
    return np.degrees(np.max(radii))

def _lloyd_relaxation(points, n_iter=100, tol=1e-10, deadline=None):
    """Lloyd relaxation: move each point toward the circumcenter of its Voronoi cell."""
    points = np.array(points, dtype=float)
    points = _normalize(points)
    n = len(points)
    
    for iteration in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute Voronoi cells using Delaunay triangulation
        try:
            delaunay = Delaunay(points)
        except Exception:
            break
        
        new_points = np.zeros_like(points)
        
        for i in range(n):
            cell_circumcenters = []
            for simplex in delaunay.simplices:
                if i in simplex:
                    facet_pts = points[simplex]
                    if len(facet_pts) == 3:
                        v1, v2, v3 = facet_pts
                        circum = np.cross(v1, v2) + np.cross(v2, v3) + np.cross(v3, v1)
                        norm = np.linalg.norm(circum)
                        if norm > 1e-15:
                            circum = circum / norm
                            cell_circumcenters.append(circum)
                    elif len(facet_pts) == 2:
                        midpoint = (facet_pts[0] + facet_pts[1])
                        norm = np.linalg.norm(midpoint)
                        if norm > 1e-15:
                            midpoint = midpoint / norm
                            cell_circumcenters.append(midpoint)
            
            if len(cell_circumcenters) > 0:
                avg = np.mean(cell_circumcenters, axis=0)
                norm = np.linalg.norm(avg)
                if norm > 1e-15:
                    new_points[i] = avg / norm
                else:
                    new_points[i] = points[i]
            else:
                new_points[i] = points[i]
        
        new_points = _normalize(new_points)
        
        diff = np.max(np.linalg.norm(new_points - points, axis=1))
        points = new_points
        
        if diff < tol:
            break
    
    return points

def _soft_max_covering_radius(points, beta=500.0):
    """Compute a smooth surrogate for the covering radius using soft-max."""
    n_samples = 2000
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    cos_dists = samples @ points.T
    cos_dists = np.clip(cos_dists, -1.0, 1.0)
    
    max_cos = np.max(cos_dists, axis=1)
    angles = np.arccos(max_cos)
    
    max_angle = np.max(angles)
    shifted = angles - max_angle
    log_sum_exp = np.log(np.sum(np.exp(beta * shifted))) + beta * max_angle
    soft_max = log_sum_exp / beta
    
    return soft_max

def _repulsion_relax(pts, n_iters=200, deadline=None):
    """Repulsion relaxation with adaptive step size."""
    pts = pts.copy()
    alpha = 0.05
    
    def repulsion_energy(p):
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        mask = ~np.eye(N, dtype=bool)
        inv_dist_sq = np.zeros_like(dist_sq)
        np.divide(1.0, dist_sq, out=inv_dist_sq, where=mask)
        return 0.5 * np.sum(inv_dist_sq)
    
    current_energy = repulsion_energy(pts)
    
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        
        mask = np.eye(N, dtype=bool)
        inv_dist_sq = np.zeros_like(dist_sq)
        np.divide(1.0, dist_sq, out=inv_dist_sq, where=~mask)
        
        forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        dot_fp = np.sum(forces * pts, axis=1)
        forces_tan = forces - dot_fp[:, np.newaxis] * pts
        
        step = alpha
        improved = False
        for _ in range(8):
            new_pts = pts + step * forces_tan
            new_pts = _normalize(new_pts)
            new_energy = repulsion_energy(new_pts)
            if new_energy < current_energy:
                pts = new_pts
                current_energy = new_energy
                improved = True
                break
            step *= 0.5
        
        if not improved:
            alpha *= 0.8
        else:
            alpha *= 1.01
        
        alpha = min(alpha, 0.1)
        alpha = max(alpha, 1e-6)
    
    return pts

def _gradient_optimize(points, deadline):
    """Gradient-based optimization on the soft-max surrogate."""
    def objective(x):
        pts = x.reshape(-1, 3)
        pts = _normalize(pts)
        return _soft_max_covering_radius(pts, beta=500.0)
    
    x0 = points.flatten()
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 100})
        if result.fun < objective(x0):
            new_points = result.x.reshape(-1, 3)
            new_points = _normalize(new_points)
            return new_points
    except Exception:
        pass
    
    return points

def _generate_initial_configurations(n_configs=8):
    """Generate multiple initial configurations for diversity."""
    configs = []
    
    for cfg_idx in range(n_configs):
        if cfg_idx == 0:
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        elif cfg_idx == 1:
            rng = np.random.RandomState(42 + cfg_idx)
            points = rng.randn(N, 3)
            points = _normalize(points)
        elif cfg_idx == 2:
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            rng = np.random.RandomState(100 + cfg_idx)
            noise = rng.randn(N, 3) * 0.1
            points = points + noise
            points = _normalize(points)
        else:
            rng = np.random.RandomState(200 + cfg_idx * 7)
            points = rng.randn(N, 3)
            points = _normalize(points)
        
        configs.append(points)
    
    return configs

def _optimize_single_config(init_points, deadline):
    """Optimize a single configuration with time budget."""
    points = init_points.copy()
    
    # Phase 1: Repulsion relaxation for better spread
    if time.time() < deadline - 30:
        points = _repulsion_relax(points, n_iters=100, deadline=deadline - 25)
    
    # Phase 2: Lloyd relaxation
    if time.time() < deadline - 15:
        points = _lloyd_relaxation(points, n_iter=50, tol=1e-8, deadline=deadline - 10)
    
    # Phase 3: Gradient optimization
    if time.time() < deadline - 5:
        points = _gradient_optimize(points, deadline=deadline - 2)
    
    # Phase 4: Final Lloyd polish
    if time.time() < deadline - 1:
        points = _lloyd_relaxation(points, n_iter=30, tol=1e-10, deadline=deadline - 0.5)
    
    return points

def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Generate initial configurations
    configs = _generate_initial_configurations(n_configs=8)
    
    best_points = None
    best_radius = np.inf
    
    for cfg_idx, init_points in enumerate(configs):
        elapsed = time.time() - start_time
        if elapsed > deadline - 20.0:
            break
        
        # Optimize this configuration
        try:
            optimized = _optimize_single_config(init_points, deadline=deadline - 5.0)
            radius = _compute_covering_radius(optimized)
            
            if radius < best_radius:
                best_radius = radius
                best_points = optimized
        except Exception:
            continue
    
    elapsed = time.time() - start_time
    if best_points is None:
        # Fallback: return the golden-angle spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Final polish with remaining time
    if time.time() < deadline - 2.0:
        try:
            final_points = _lloyd_relaxation(best_points, n_iter=50, tol=1e-12, deadline=deadline - 1.0)
            final_radius = _compute_covering_radius(final_points)
            if final_radius < best_radius:
                best_points = final_points
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
