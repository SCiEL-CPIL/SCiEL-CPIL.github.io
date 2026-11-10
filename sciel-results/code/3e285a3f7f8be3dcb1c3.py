import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return points / norms

def _covering_radius(points):
    """Calculate the covering radius in radians for points on the unit sphere."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi

    simplices = hull.simplices
    tri_pts = points[simplices]
    
    p0 = tri_pts[:, 0, :]
    p1 = tri_pts[:, 1, :]
    p2 = tri_pts[:, 2, :]
    
    v1 = p1 - p0
    v2 = p2 - p0
    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1)
    
    mask = norms > 1e-12
    if not np.any(mask):
        return 0.0
        
    d = np.zeros(len(normals))
    d[mask] = np.abs(np.sum(normals[mask] * p0[mask], axis=1)) / norms[mask]
    d = np.clip(d, 0.0, 1.0)
    alphas = np.arccos(d)
    return np.max(alphas)

def _smooth_objective(params, beta=50.0):
    """Smooth surrogate for the covering radius using log-sum-exp."""
    pts = _normalize(params.reshape(N, 3))
    if not np.all(np.isfinite(pts)):
        return 1e6
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi
    simplices = hull.simplices
    tri_pts = pts[simplices]
    p0 = tri_pts[:, 0, :]
    p1 = tri_pts[:, 1, :]
    p2 = tri_pts[:, 2, :]
    v1 = p1 - p0
    v2 = p2 - p0
    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1)
    mask = norms > 1e-12
    if not np.any(mask):
        return 0.0
    d = np.zeros(len(normals))
    d[mask] = np.abs(np.sum(normals[mask] * p0[mask], axis=1)) / norms[mask]
    d = np.clip(d, 0.0, 1.0)
    alphas = np.arccos(d)
    max_a = np.max(alphas)
    return max_a + np.log(np.sum(np.exp(beta * (alphas - max_a)))) / beta

def _circumcenter_polish(points, max_iter=200, tol=1e-10, time_limit=None, start_time=None):
    """Polish by moving each point toward the circumcenter of its largest adjacent facet."""
    if start_time is None:
        start_time = time.time()
    points = points.copy()
    n = points.shape[0]
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        simplices = hull.simplices
        tri_pts = points[simplices]
        p0 = tri_pts[:, 0, :]
        p1 = tri_pts[:, 1, :]
        p2 = tri_pts[:, 2, :]
        v1 = p1 - p0
        v2 = p2 - p0
        normals = np.cross(v1, v2)
        norms = np.linalg.norm(normals, axis=1)
        mask = norms > 1e-12
        if not np.any(mask):
            break
        d = np.zeros(len(normals))
        d[mask] = np.abs(np.sum(normals[mask] * p0[mask], axis=1)) / norms[mask]
        d = np.clip(d, 0.0, 1.0)
        alphas = np.arccos(d)
        
        updates = np.zeros_like(points)
        counts = np.zeros(n)
        
        for i, simplex in enumerate(simplices):
            if not mask[i]:
                continue
            weight = np.exp(50 * (alphas[i] - np.max(alphas[mask])))
            for vertex_idx in simplex:
                centroid = (p0[i] + p1[i] + p2[i]) / 3.0
                circumcenter = centroid / np.linalg.norm(centroid)
                updates[vertex_idx] += weight * (circumcenter - points[vertex_idx])
                counts[vertex_idx] += weight
        
        for i in range(n):
            if counts[i] > 1e-10:
                updates[i] /= counts[i]
        
        step_size = 0.1
        points = points + step_size * updates
        points = _normalize(points)
        
        if np.max(np.abs(updates)) < tol:
            break
    
    return points

def _repulsive_relaxation(points, steps=500, alpha=0.1, time_limit=None, start_time=None):
    """Relax points using repulsive forces to spread them out."""
    if start_time is None:
        start_time = time.time()
    points = points.copy()
    n = points.shape[0]
    
    for step in range(steps):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        mask = dist_sq > 1e-12
        inv_dist3 = np.zeros_like(dist_sq)
        inv_dist3[mask] = 1.0 / (dist_sq[mask] * np.sqrt(dist_sq[mask]))
        forces = np.sum(diff * inv_dist3[:, :, np.newaxis], axis=1)
        points += alpha * forces
        points = _normalize(points)
    
    return points

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    def time_remaining():
        return deadline - time.time()

    # 1. Generate diverse initial configurations
    fib_points = _fibonacci_sphere(N)
    
    rng = np.random.RandomState(42)
    init_configs = [fib_points.copy()]
    
    # Antipodal pairs + random
    for restart in range(4):
        pts = rng.randn(N, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        # Add antipodal perturbation
        for _ in range(10):
            idx = rng.randint(N)
            pts[idx] = -pts[idx]
            pts = _normalize(pts)
        init_configs.append(pts.copy())
    
    # Multi-scale perturbations of Fibonacci
    for scale in [0.1, 0.3, 0.5]:
        perturbed = fib_points + scale * rng.randn(N, 3)
        init_configs.append(_normalize(perturbed))
    
    # Relax each and pick the best
    best_points = None
    best_radius = np.inf
    
    for cfg in init_configs:
        if time_remaining() < 5:
            break
        relaxed = _repulsive_relaxation(cfg, steps=300, alpha=0.05, time_limit=3, start_time=start_time)
        rad = _covering_radius(relaxed)
        if rad < best_radius:
            best_radius = rad
            best_points = relaxed.copy()
    
    # 2. Adaptive beta annealing optimization
    def optimize_from(points, start_time_ref):
        nonlocal best_points, best_radius
        
        # Phase 1: L-BFGS with smooth surrogate, annealing beta
        betas = [5.0, 10.0, 20.0, 50.0, 100.0]
        current = points.copy()
        
        for beta in betas:
            if time_remaining() < 10:
                break
            
            def obj(params):
                if time.time() > deadline:
                    return 1e6
                return _smooth_objective(params, beta=beta)
            
            try:
                result = minimize(obj, current.flatten(), method='L-BFGS-B',
                                  options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 500})
                if result.fun < 1e5:
                    new_pts = _normalize(result.x.reshape(N, 3))
                    rad = _covering_radius(new_pts)
                    if rad < best_radius:
                        best_radius = rad
                        best_points = new_pts.copy()
                    current = new_pts.copy()
            except Exception:
                pass
        
        # Phase 2: Circumcenter polish
        if time_remaining() > 8:
            polished = _circumcenter_polish(current, max_iter=150, time_limit=4, start_time=start_time_ref)
            rad = _covering_radius(polished)
            if rad < best_radius:
                best_radius = rad
                best_points = polished.copy()
                current = polished.copy()
        
        # Phase 3: Nelder-Mead with exact objective
        if time_remaining() > 6:
            def exact_obj(params):
                if time.time() > deadline:
                    return 1e6
                pts = _normalize(params.reshape(N, 3))
                if not np.all(np.isfinite(pts)):
                    return 1e6
                return _covering_radius(pts)
            
            try:
                result_nm = minimize(exact_obj, current.flatten(), method='Nelder-Mead',
                                     options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
                if result_nm.fun < best_radius:
                    best_points = _normalize(result_nm.x.reshape(N, 3))
                    best_radius = result_nm.fun
                    current = best_points.copy()
            except Exception:
                pass
        
        return current

    current = optimize_from(best_points, start_time)
    
    # 3. Random restarts with diverse perturbations
    while time_remaining() > 20:
        noise_scales = [0.15, 0.25, 0.4]
        for scale in noise_scales:
            if time_remaining() < 15:
                break
            noise = rng.normal(0, scale, size=best_points.shape)
            perturbed = _normalize(best_points + noise)
            
            # Quick relaxation
            relaxed = _repulsive_relaxation(perturbed, steps=100, alpha=0.05, 
                                            time_limit=2, start_time=start_time)
            rad = _covering_radius(relaxed)
            
            if rad < best_radius:
                best_points = relaxed
                best_radius = rad
                current = optimize_from(best_points, start_time)
            elif time_remaining() > 25:
                small_polish = _circumcenter_polish(perturbed, max_iter=50, time_limit=2, start_time=start_time)
                rad2 = _covering_radius(small_polish)
                if rad2 < best_radius:
                    best_points = small_polish
                    best_radius = rad2
                    current = optimize_from(best_points, start_time)
    
    # 4. Final high-precision polish
    if time_remaining() > 8:
        polished = _circumcenter_polish(best_points, max_iter=300, time_limit=5, start_time=start_time)
        rad = _covering_radius(polished)
        if rad < best_radius:
            best_points = polished
            best_radius = rad
        
        if time_remaining() > 4:
            def exact_obj(params):
                if time.time() > deadline:
                    return 1e6
                pts = _normalize(params.reshape(N, 3))
                if not np.all(np.isfinite(pts)):
                    return 1e6
                return _covering_radius(pts)
            
            try:
                result_final = minimize(exact_obj, best_points.flatten(), method='Nelder-Mead',
                                        options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True})
                if result_final.fun < best_radius:
                    best_points = _normalize(result_final.x.reshape(N, 3))
                    best_radius = result_final.fun
            except Exception:
                pass

    return best_points
