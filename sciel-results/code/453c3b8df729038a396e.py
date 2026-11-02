import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.zeros(3)
    return v / n

def _fib_sample(n, phase=0.5):
    i = np.arange(n) + phase
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_hull_data(p):
    try:
        hull = ConvexHull(p)
        facets = hull.simplices
    except Exception:
        return None, None, None
    
    n_facets = len(facets)
    if n_facets == 0:
        return None, None, None
    
    a = p[facets[:, 0]]
    b = p[facets[:, 1]]
    c = p[facets[:, 2]]
    
    v1 = a - b
    v2 = b - c
    n = np.cross(v1, v2)
    norm_n = np.linalg.norm(n, axis=1)
    
    valid = norm_n > 1e-12
    circumcenters = np.zeros((n_facets, 3))
    radii = np.zeros(n_facets)
    
    if np.any(valid):
        cc = n[valid] / norm_n[valid, np.newaxis]
        dots = np.sum(cc * a[valid], axis=1)
        cc[dots < 0] *= -1
        circumcenters[valid] = cc
        
        cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
        radii[valid] = np.arccos(cos_r)
    
    return facets, circumcenters, radii

def _covering_radius_deg_hull(config):
    facets, circumcenters, radii = _compute_hull_data(config)
    if radii is None:
        grid = _fib_sample(20000)
        dots = grid @ config.T
        min_dots = np.min(dots, axis=1)
        max_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
        return np.degrees(np.max(max_angle))
    return np.degrees(np.max(radii))

def _quadrature_objective(params, n_grid=5000):
    """
    Smooth objective: mean of 1/(1 - cos(d_i)) where d_i is angular distance to nearest site.
    This penalizes large empty caps smoothly and is differentiable.
    """
    X = params.reshape(N, 3)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    X = X / norms
    
    # Dense Fibonacci grid for quadrature
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # For each grid point, find nearest site
    dots = grid @ X.T  # (n_grid, N)
    min_dots = np.min(dots, axis=1)  # (n_grid,)
    
    # 1/(1 - cos d) = 1/(1 - dot) where dot = cos d
    # Clip to avoid singularity
    denom = np.clip(1.0 - min_dots, 1e-6, None)
    vals = 1.0 / denom
    
    return np.mean(vals)

def _quadrature_grad(params, n_grid=3000, eps=1e-5):
    """
    Numerical gradient of quadrature objective.
    """
    X = params.reshape(N, 3)
    grad = np.zeros_like(X)
    
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    for k in range(N * 3):
        ki = k // 3
        kj = k % 3
        
        X_plus = X.copy()
        X_plus[ki, kj] += eps
        n_plus = np.linalg.norm(X_plus[ki])
        X_plus[ki] = X_plus[ki] / n_plus if n_plus > 1e-12 else X_plus[ki]
        
        dots_plus = grid @ X_plus.T
        min_dots_plus = np.min(dots_plus, axis=1)
        denom_plus = np.clip(1.0 - min_dots_plus, 1e-6, None)
        val_plus = np.mean(1.0 / denom_plus)
        
        X_minus = X.copy()
        X_minus[ki, kj] -= eps
        n_minus = np.linalg.norm(X_minus[ki])
        X_minus[ki] = X_minus[ki] / n_minus if n_minus > 1e-12 else X_minus[ki]
        
        dots_minus = grid @ X_minus.T
        min_dots_minus = np.min(dots_minus, axis=1)
        denom_minus = np.clip(1.0 - min_dots_minus, 1e-6, None)
        val_minus = np.mean(1.0 / denom_minus)
        
        grad[ki, kj] = (val_plus - val_minus) / (2.0 * eps)
    
    return grad

def _repulsion_relax(points, alpha_init=0.1, decay=0.99, max_iter=500, deadline=None):
    """
    Repulsion relaxation with adaptive step size.
    Minimize sum 1/||p_i - p_j||^2 to spread points evenly.
    """
    points = points.copy()
    alpha = alpha_init
    best_score = _covering_radius_deg_hull(points)
    best_pts = points.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        
        mask = np.eye(N, dtype=bool)
        inv_dist_sq = np.zeros_like(dist_sq)
        np.divide(1.0, dist_sq, out=inv_dist_sq, where=~mask)
        
        forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        dot_fp = np.sum(forces * points, axis=1)
        forces_tan = forces - dot_fp[:, np.newaxis] * points
        
        points = points + alpha * forces_tan
        
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]
        
        alpha *= decay
        
        if it % 20 == 0:
            score = _covering_radius_deg_hull(points)
            if score < best_score:
                best_score = score
                best_pts = points.copy()
    
    return best_pts

def _soft_max_objective(params, beta=100.0):
    X = params.reshape(N, 3)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    X = X / norms
    
    facets, circumcenters, radii = _compute_hull_data(X)
    if facets is None:
        return 1e10
        
    r_max = np.max(radii)
    if r_max < 1e-15:
        return r_max
    shifted = beta * (radii - r_max)
    log_sum_exp = np.log(np.sum(np.exp(shifted))) / beta + r_max
    return log_sum_exp

def _soft_max_grad(params, beta=100.0, eps=1e-5):
    """
    Numerical gradient of soft-max objective.
    """
    X = params.reshape(N, 3)
    grad = np.zeros_like(X)
    
    for k in range(N * 3):
        ki = k // 3
        kj = k % 3
        
        X_plus = X.copy()
        X_plus[ki, kj] += eps
        n_plus = np.linalg.norm(X_plus[ki])
        if n_plus > 1e-12:
            X_plus[ki] = X_plus[ki] / n_plus
        val_plus = _soft_max_objective(X_plus.flatten(), beta)
        
        X_minus = X.copy()
        X_minus[ki, kj] -= eps
        n_minus = np.linalg.norm(X_minus[ki])
        if n_minus > 1e-12:
            X_minus[ki] = X_minus[ki] / n_minus
        val_minus = _soft_max_objective(X_minus.flatten(), beta)
        
        grad[ki, kj] = (val_plus - val_minus) / (2.0 * eps)
    
    return grad

def _lbfgs_stage(start_pts, beta, deadline, maxiter=200, use_grad=True):
    """
    Run L-BFGS-B with soft-max objective at a given beta.
    """
    try:
        if use_grad:
            res = minimize(_soft_max_objective, start_pts.flatten(), method='L-BFGS-B',
                           jac=lambda x: _soft_max_grad(x, beta).flatten(),
                           options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 300},
                           args=(beta,))
        else:
            res = minimize(_soft_max_objective, start_pts.flatten(), method='L-BFGS-B',
                           options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 300},
                           args=(beta,))
        
        new_pts = res.x.reshape(N, 3)
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        new_pts = new_pts / norms
        
        return new_pts
    except Exception:
        return start_pts

def _quadrature_lbfgs(start_pts, deadline, maxiter=100):
    """
    Run L-BFGS-B on the quadrature (mean reciprocal distance) objective.
    """
    try:
        res = minimize(_quadrature_objective, start_pts.flatten(), method='L-BFGS-B',
                       jac=lambda x: _quadrature_grad(x).flatten(),
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 200},
                       args=(3000,))
        
        new_pts = res.x.reshape(N, 3)
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        new_pts = new_pts / norms
        
        return new_pts
    except Exception:
        return start_pts

def _facet_local_search(config, deadline, max_iter=200, step=0.02):
    """
    Direct local search targeting the worst facet.
    For the facet with maximum circumradius, move its vertices to reduce the radius.
    """
    config = config.copy()
    best_config = config.copy()
    best_score = _covering_radius_deg_hull(config)
    s = step
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        facets, circumcenters, radii = _compute_hull_data(config)
        if facets is None:
            break
        
        # Find the worst facet
        max_idx = np.argmax(radii)
        max_facet = facets[max_idx]
        cc = circumcenters[max_idx]
        r_max = radii[max_idx]
        
        if r_max < 0.32:  # ~18.3 degrees in radians
            break
        
        # Try moving each vertex of the worst facet
        improved = False
        for vi in max_facet:
            # Direction: move vertex away from circumcenter (projected to tangent)
            dir_vec = config[vi] - cc
            dir_norm = np.linalg.norm(dir_vec)
            if dir_norm < 1e-10:
                continue
            dir_vec = dir_vec / dir_norm
            
            # Project to tangent plane
            dot = np.dot(dir_vec, config[vi])
            dir_tan = dir_vec - dot * config[vi]
            dir_norm_tan = np.linalg.norm(dir_tan)
            if dir_norm_tan < 1e-10:
                continue
            dir_tan = dir_tan / dir_norm_tan
            
            # Try several step sizes
            for trial_s in [s, s * 0.5, s * 0.25]:
                new_config = config.copy()
                new_config[vi] = _normalize(config[vi] + trial_s * dir_tan)
                
                new_score = _covering_radius_deg_hull(new_config)
                if new_score < best_score - 1e-10:
                    best_score = new_score
                    best_config = new_config.copy()
                    config = new_config
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try random perturbations of the worst facet vertices
            for vi in max_facet:
                for trial_s in [s, s * 0.5]:
                    rand_dir = np.random.randn(3)
                    dot = np.dot(rand_dir, config[vi])
                    rand_dir = rand_dir - dot * config[vi]
                    rand_norm = np.linalg.norm(rand_dir)
                    if rand_norm < 1e-10:
                        continue
                    rand_dir = rand_dir / rand_norm
                    
                    new_config = config.copy()
                    new_config[vi] = _normalize(config[vi] + trial_s * rand_dir)
                    
                    new_score = _covering_radius_deg_hull(new_config)
                    if new_score < best_score - 1e-10:
                        best_score = new_score
                        best_config = new_config.copy()
                        config = new_config
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            s *= 0.8
            if s < 1e-5:
                break
    
    return best_config

def _lloyd_step(config):
    """
    One Lloyd iteration: move each point toward the spherical mean of 
    circumcenters of facets for which it is the nearest site.
    """
    try:
        hull = ConvexHull(config)
    except Exception:
        return config
    
    facets = hull.simplices
    a = config[facets[:, 0]]
    b = config[facets[:, 1]]
    c = config[facets[:, 2]]
    
    v1 = a - b
    v2 = b - c
    n = np.cross(v1, v2)
    norm_n = np.linalg.norm(n, axis=1)
    
    valid = norm_n > 1e-12
    if not np.any(valid):
        return config
    
    cc = n[valid] / norm_n[valid, np.newaxis]
    dots = np.sum(cc * a[valid], axis=1)
    cc[dots < 0] *= -1
    
    dots_all = cc @ config.T
    nearest_idx = np.argmax(dots_all, axis=1)
    
    sums = np.zeros((N, 3))
    counts = np.zeros(N)
    
    np.add.at(sums, nearest_idx, cc)
    np.add.at(counts, nearest_idx, 1)
    
    new_config = config.copy()
    for i in range(N):
        if counts[i] > 0:
            target = sums[i]
            n = np.linalg.norm(target)
            if n > 1e-10:
                new_config[i] = target / n
            else:
                new_config[i] = config[i]
        else:
            new_config[i] = config[i]
            
    return new_config

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Fibonacci spiral variants
    for phase in [0.5, 0.37, 0.618, 0.25, 0.75, 0.1, 0.9]:
        initial_configs.append(_fib_sample(N, phase=phase))
    
    # Random points
    for seed in range(5):
        np.random.seed(100 + seed * 17)
        rand_pts = np.random.randn(N, 3)
        rand_pts = np.array([_normalize(p) for p in rand_pts])
        initial_configs.append(rand_pts)
    
    best_config = initial_configs[0].copy()
    best_score = _covering_radius_deg_hull(best_config)
    
    # Phase 1: Repulsion relaxation from multiple starts
    for config in initial_configs:
        if time.time() > deadline - 60.0:
            break
        relaxed = _repulsion_relax(config, alpha_init=0.1, decay=0.99, max_iter=400, deadline=deadline - 60.0)
        score = _covering_radius_deg_hull(relaxed)
        if score < best_score:
            best_score = score
            best_config = relaxed.copy()
    
    # Phase 2: Quadrature-based optimization (smooth global signal)
    if time.time() < deadline - 35.0:
        q_pts = _quadrature_lbfgs(best_config, deadline - 35.0, maxiter=80)
        q_score = _covering_radius_deg_hull(q_pts)
        if q_score < best_score:
            best_score = q_score
            best_config = q_pts.copy()
    
    # Phase 3: Multi-scale soft-max L-BFGS with analytical gradient
    current_pts = best_config.copy()
    betas = [20.0, 50.0, 100.0, 200.0, 400.0, 800.0]
    
    for beta in betas:
        if time.time() > deadline - 20.0:
            break
        
        new_pts = _lbfgs_stage(current_pts, beta, deadline - 20.0, maxiter=150, use_grad=True)
        new_score = _covering_radius_deg_hull(new_pts)
        
        if new_score < best_score:
            best_score = new_score
            best_config = new_pts.copy()
            current_pts = new_pts.copy()
    
    # Phase 4: Lloyd's algorithm
    if time.time() < deadline - 15.0:
        config = best_config.copy()
        best_lloyd = config.copy()
        best_lloyd_score = _covering_radius_deg_hull(config)
        
        it = 0
        while time.time() < deadline - 15.0 and it < 300:
            new_config = _lloyd_step(config)
            new_score = _covering_radius_deg_hull(new_config)
            
            if new_score < best_lloyd_score - 1e-10:
                best_lloyd_score = new_score
                best_lloyd = new_config.copy()
                config = new_config
            else:
                # Perturb
                idx = np.random.randint(N)
                rand_dir = np.random.randn(3)
                dot = np.dot(rand_dir, config[idx])
                rand_dir = rand_dir - dot * config[idx]
                rand_dir = _normalize(rand_dir)
                config[idx] = _normalize(config[idx] + 0.03 * rand_dir)
                pert_score = _covering_radius_deg_hull(config)
                if pert_score < best_lloyd_score:
                    best_lloyd_score = pert_score
                    best_lloyd = config.copy()
                else:
                    config[idx] = best_lloyd[idx]
            
            it += 1
        
        if best_lloyd_score < best_score:
            best_score = best_lloyd_score
            best_config = best_lloyd.copy()
    
    # Phase 5: Facet-based local search
    if time.time() < deadline - 5.0:
        polished = _facet_local_search(best_config, deadline - 5.0, max_iter=150, step=0.03)
        pol_score = _covering_radius_deg_hull(polished)
        if pol_score < best_score:
            best_score = pol_score
            best_config = polished.copy()
    
    # Phase 6: Final high-precision L-BFGS polish
    if time.time() < deadline - 2.0:
        final = _lbfgs_stage(best_config, 1000.0, deadline - 2.0, maxiter=100, use_grad=True)
        final_score = _covering_radius_deg_hull(final)
        if final_score < best_score:
            best_score = final_score
            best_config = final.copy()
    
    # Final normalization and validity check
    best_config = np.array([_normalize(p) for p in best_config])
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_config[i] - best_config[j]) < 1e-6:
                best_config[j] = best_config[j] + 1e-4 * np.random.randn(3)
                best_config[j] = _normalize(best_config[j])
    
    return best_config
