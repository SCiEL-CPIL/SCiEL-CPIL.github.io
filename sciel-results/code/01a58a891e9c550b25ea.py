import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _normalize(v):
    """Normalize vector to unit length."""
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.zeros(3)
    return v / n

def _fib_sample(n, phase=0.5):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + phase
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_hull_data(p):
    """
    Compute convex hull facets and their circumcenters/radii (vectorized).
    Returns facets, circumcenters, radii.
    """
    try:
        hull = ConvexHull(p)
        facets = hull.simplices
    except Exception:
        return None, None, None
    
    n_facets = len(facets)
    if n_facets == 0:
        return None, None, None
    
    a = p[facets[:, 0]]  # (F, 3)
    b = p[facets[:, 1]]
    c = p[facets[:, 2]]
    
    v1 = a - b  # (F, 3)
    v2 = b - c
    n = np.cross(v1, v2)  # (F, 3)
    norm_n = np.linalg.norm(n, axis=1)  # (F,)
    
    valid = norm_n > 1e-12
    circumcenters = np.zeros((n_facets, 3))
    radii = np.zeros(n_facets)
    
    if np.any(valid):
        cc = n[valid] / norm_n[valid, np.newaxis]
        # Ensure circumcenter faces outward (toward 'a')
        dots = np.sum(cc * a[valid], axis=1)
        cc[dots < 0] *= -1
        circumcenters[valid] = cc
        
        cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
        radii[valid] = np.arccos(cos_r)
    
    return facets, circumcenters, radii

def _covering_radius_deg_hull(config):
    """
    Compute the exact covering radius in degrees using convex hull facets.
    """
    facets, circumcenters, radii = _compute_hull_data(config)
    if radii is None:
        # Fallback to sampling if hull fails
        grid = _fib_sample(20000)
        dots = grid @ config.T
        min_dots = np.min(dots, axis=1)
        max_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
        return np.degrees(np.max(max_angle))
    
    return np.degrees(np.max(radii))

def _epigraph_objective(params, beta=50.0):
    """
    Epigraph formulation with soft-max.
    params: [x_1, ..., x_{3N}, t] where t is the auxiliary variable.
    Objective: t + (1/beta) * log(sum(exp(beta * (r_i - t))))
    This is smooth and equals max(r_i) as beta -> inf.
    """
    X = params[:3*N].reshape(N, 3)
    t = params[3*N]
    
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    X = X / norms
    
    facets, circumcenters, radii = _compute_hull_data(X)
    if facets is None:
        return 1e10
    
    # Soft max of (radii - t)
    r_shifted = radii - t
    r_max_shifted = np.max(r_shifted)
    if r_max_shifted < -100:
        return t  # All radii well below t
    
    shifted = beta * (r_shifted - r_max_shifted)
    log_sum_exp = np.log(np.sum(np.exp(shifted))) / beta + r_max_shifted
    
    return t + log_sum_exp

def _soft_max_objective(params, beta=100.0):
    """
    Soft-max surrogate: approximates max(radii) via log-sum-exp.
    As beta -> inf, this approaches the true max.
    """
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

def _repulsion_relax(points, alpha_init=0.05, decay=0.995, max_iter=300, deadline=None):
    """
    Repulsion Relaxation (Thomson Problem solver).
    Minimize sum 1/||p_i - p_j||^2 to spread points evenly.
    """
    points = points.copy()
    alpha = alpha_init
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Calculate forces
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        
        # Avoid division by zero
        mask = np.eye(N, dtype=bool)
        inv_dist_sq = np.zeros_like(dist_sq)
        np.divide(1.0, dist_sq, out=inv_dist_sq, where=~mask)
        
        # Force F_i = sum_j (p_i - p_j) / ||p_i - p_j||^2
        forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        # Project onto tangent plane: F_tan = F - (F . p) p
        dot_fp = np.sum(forces * points, axis=1)
        forces_tan = forces - dot_fp[:, np.newaxis] * points
        
        # Update
        points = points + alpha * forces_tan
        
        # Normalize
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]
        
        # Decay alpha
        alpha *= decay
        
    return points

def _lloyd_step_vectorized(config):
    """
    Perform one Lloyd iteration (vectorized).
    Moves each point towards the spherical mean of the circumcenters of the faces for which it is the nearest site.
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
    
    # For each circumcenter, find the nearest site
    dots_all = cc @ config.T
    nearest_idx = np.argmax(dots_all, axis=1)
    
    # Accumulate targets for each site
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

def _facet_based_local_search(config, deadline, max_iter=200):
    """
    Direct facet-based local search: move vertices of the largest facet
    to reduce its circumradius.
    """
    config = config.copy()
    best_config = config.copy()
    best_score = _covering_radius_deg_hull(config)
    
    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        
        facets, circumcenters, radii = _compute_hull_data(config)
        if facets is None:
            break
        
        max_idx = np.argmax(radii)
        max_facet = facets[max_idx]
        cc = circumcenters[max_idx]
        
        # Try moving each vertex of the max facet
        improved = False
        for vi in max_facet:
            # Direction to shrink: move vertex toward circumcenter
            dir_vec = cc - config[vi]
            dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-15)
            # Project to tangent plane
            dot = np.dot(dir_vec, config[vi])
            dir_tan = dir_vec - dot * config[vi]
            dir_tan = dir_tan / (np.linalg.norm(dir_tan) + 1e-15)
            
            # Try several step sizes
            for step in [0.02, 0.01, 0.005, 0.001]:
                new_config = config.copy()
                new_config[vi] = _normalize(config[vi] + step * dir_tan)
                new_score = _covering_radius_deg_hull(new_config)
                
                if new_score < best_score - 1e-12:
                    best_score = new_score
                    best_config = new_config.copy()
                    config = new_config
                    improved = True
                    break
        
        if not improved:
            # Try random perturbations to escape local minimum
            idx = np.random.randint(N)
            rand_dir = np.random.randn(3)
            rand_dir = rand_dir - np.dot(rand_dir, config[idx]) * config[idx]
            rand_dir = _normalize(rand_dir)
            old_pt = config[idx].copy()
            config[idx] = _normalize(config[idx] + 0.03 * rand_dir)
            pert_score = _covering_radius_deg_hull(config)
            if pert_score < best_score:
                best_score = pert_score
                best_config = config.copy()
            else:
                config[idx] = old_pt
                # Check if we've converged
                if it > 50:
                    break
    
    return best_config, best_score

def _high_precision_polish(config, deadline):
    """
    High-precision final polish using L-BFGS with very tight tolerances
    and high beta for near-exact max.
    """
    config = config.copy()
    best_config = config.copy()
    best_score = _covering_radius_deg_hull(config)
    
    # Use epigraph formulation for smoother optimization
    initial_params = np.concatenate([config.flatten(), [best_score * np.pi / 180.0]])
    
    for beta in [200.0, 500.0, 1000.0]:
        if time.time() > deadline - 2.0:
            break
        
        try:
            res = minimize(_epigraph_objective, initial_params, method='L-BFGS-B',
                           options={'maxiter': 150, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 200},
                           args=(beta,))
            
            new_pts = res.x[:3*N].reshape(N, 3)
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            new_pts = new_pts / norms
            
            current_radius = _covering_radius_deg_hull(new_pts)
            if current_radius < best_score:
                best_score = current_radius
                best_config = new_pts.copy()
                initial_params = np.concatenate([new_pts.flatten(), [best_score * np.pi / 180.0]])
        except Exception:
            pass
    
    return best_config, best_score

def _simulated_annealing(config, deadline, T_init=0.1, T_min=0.001, cool_rate=0.98, max_iter=500):
    """
    Simulated annealing with facet-based evaluation.
    """
    config = config.copy()
    best_config = config.copy()
    best_score = _covering_radius_deg_hull(config)
    T = T_init
    rng = np.random.default_rng(42)
    
    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        
        # Pick a random point to perturb
        idx = rng.integers(N)
        
        # Generate random perturbation in tangent plane
        rand_dir = rng.standard_normal(3)
        rand_dir = rand_dir - np.dot(rand_dir, config[idx]) * config[idx]
        rand_dir = _normalize(rand_dir)
        
        # Perturb magnitude
        step = 0.02 * (1 + rng.standard_normal())
        new_point = _normalize(config[idx] + step * rand_dir)
        
        old_point = config[idx].copy()
        config[idx] = new_point
        
        new_score = _covering_radius_deg_hull(config)
        delta = new_score - best_score
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or (T > T_min and rng.random() < np.exp(-delta / T)):
            if new_score < best_score:
                best_score = new_score
                best_config = config.copy()
        else:
            config[idx] = old_point
        
        T *= cool_rate
    
    return best_config, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # 1. Fibonacci spiral
    fib_pts = _fib_sample(N)
    initial_configs.append(fib_pts)
    
    # 2. Fibonacci with different phase offsets
    for phase in [0.37, 0.618, 0.25, 0.12, 0.88]:
        i2 = np.arange(N) + 0.5
        z2 = 1.0 - 2.0 * i2 / N
        phi2 = np.pi * (1.0 + np.sqrt(5.0)) * (i2 + phase)
        r2 = np.sqrt(1.0 - z2 * z2)
        fib2_pts = np.column_stack([r2 * np.cos(phi2), r2 * np.sin(phi2), z2])
        initial_configs.append(fib2_pts)
    
    # 3. Random points (seeded for reproducibility)
    for seed_offset in range(4):
        np.random.seed(42 + seed_offset)
        rand_pts = np.random.randn(N, 3)
        rand_pts = np.array([_normalize(p) for p in rand_pts])
        initial_configs.append(rand_pts)
    
    best_config = initial_configs[0].copy()
    best_score = _covering_radius_deg_hull(best_config)
    
    # Phase 1: Repulsion Relaxation to spread points
    for config in initial_configs:
        if time.time() > deadline - 60.0:
            break
        relaxed = _repulsion_relax(config, alpha_init=0.05, decay=0.995, max_iter=300, deadline=deadline - 60.0)
        score = _covering_radius_deg_hull(relaxed)
        if score < best_score:
            best_score = score
            best_config = relaxed.copy()
    
    # Phase 2: L-BFGS optimization with soft-max objective
    starting_points = [best_config.copy(), _fib_sample(N)]
    
    for start_pts in starting_points:
        if time.time() > deadline - 30.0:
            break
        
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 20.0:
                break
            
            try:
                res = minimize(_soft_max_objective, start_pts.flatten(), method='L-BFGS-B',
                               options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200},
                               args=(beta,))
                
                new_pts = res.x.reshape(N, 3)
                norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
                norms[norms < 1e-12] = 1.0
                new_pts = new_pts / norms
                
                current_radius = _covering_radius_deg_hull(new_pts)
                if current_radius < best_score:
                    best_score = current_radius
                    best_config = new_pts.copy()
            except Exception:
                pass
    
    # Phase 3: Facet-based local search (directly targets max circumradius)
    if time.time() < deadline - 25.0:
        fbs_config, fbs_score = _facet_based_local_search(best_config, deadline - 25.0, max_iter=300)
        if fbs_score < best_score:
            best_score = fbs_score
            best_config = fbs_config.copy()
    
    # Phase 4: Simulated annealing for global search
    if time.time() < deadline - 15.0:
        sa_config, sa_score = _simulated_annealing(best_config, deadline - 15.0, T_init=0.08, T_min=0.001, cool_rate=0.985, max_iter=300)
        if sa_score < best_score:
            best_score = sa_score
            best_config = sa_config.copy()
    
    # Phase 5: High-precision polish with epigraph formulation
    if time.time() < deadline - 5.0:
        polished, final_score = _high_precision_polish(best_config, deadline - 5.0)
        if final_score < best_score:
            best_score = final_score
            best_config = polished.copy()
    
    # Final normalization
    best_config = np.array([_normalize(p) for p in best_config])
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_config[i] - best_config[j]) < 1e-6:
                best_config[j] = best_config[j] + 1e-4 * np.random.randn(3)
                best_config[j] = _normalize(best_config[j])
                
    return best_config
