"""Packing 50 equal circles in a unit circle using smooth optimization and simulated annealing."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # safety margin before 150s kill


def _score_vec(points):
    """Vectorized score computation."""
    points = np.asarray(points).reshape(-1, 2)
    n = len(points)
    radii = 1.0 - np.linalg.norm(points, axis=1)
    r = np.min(radii)
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Only consider i < j
    iu = np.triu_indices(n, 1)
    pair_dists = dists[iu] / 2.0
    r = min(r, np.min(pair_dists))
    return r


def _ring_init(n, n_rings=5):
    """Generate a ring-based initial configuration with concentric circles."""
    best_points = None
    best_r = 0.0
    
    # Try different ring configurations
    ring_configs = [
        [1, 6, 12, 18, 13],  # Center + 4 rings
        [1, 6, 12, 20, 11],  # Center + 4 rings
        [1, 6, 14, 18, 11],  # Center + 4 rings
        [1, 6, 12, 18, 13],  # Same as first for now
        [6, 12, 18, 14],     # No center, 4 rings
    ]
    
    for config in ring_configs:
        if sum(config) != n:
            continue
            
        points = []
        # Add center if first ring has 1
        if config[0] == 1:
            points.append([0.0, 0.0])
            config_remaining = config[1:]
        else:
            config_remaining = config
            
        for ring_idx, count in enumerate(config_remaining):
            # Radius of this ring, spaced to avoid overlap
            ring_radius = (ring_idx + 1) * 0.18
            for k in range(count):
                angle = 2 * np.pi * k / count + ring_idx * np.pi / count
                x = ring_radius * np.cos(angle)
                y = ring_radius * np.sin(angle)
                points.append([x, y])
        
        points = np.array(points)
        if len(points) == n:
            r = _score_vec(points)
            if r > best_r:
                best_r = r
                best_points = points.copy()
    
    if best_points is None:
        # Fallback to hexagonal
        best_points = _hexagonal_init(n)
    
    return best_points


def _hexagonal_init(n):
    """Generate a hexagonal lattice initial configuration, picking best spacing."""
    best_points = None
    best_r = 0.0
    for spacing in np.linspace(0.05, 0.25, 50):
        s = spacing
        points = []
        max_coord = int(1.0 / s) + 2
        for j in range(-max_coord, max_coord + 1):
            for i in range(-max_coord, max_coord + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3) / 2
                if np.hypot(x, y) <= 1.0 - s / 2:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= n:
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:n]
            cand = points[idx]
            r = _score_vec(cand)
            if r > best_r:
                best_r = r
                best_points = cand.copy()
    if best_points is None:
        best_points = np.random.randn(n, 2) * 0.5
    return best_points


def _smooth_objective(params, b=50.0):
    """
    Smooth surrogate for maximizing the minimum of boundary and pairwise distances.
    Uses log-sum-exp to approximate the min function smoothly for L-BFGS-B.
    """
    pts = params.reshape(-1, 2)
    n = len(pts)
    
    # Boundary distances
    boundary = 1.0 - np.linalg.norm(pts, axis=1)
    
    # Pairwise half-distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, 1)
    pair_dists = dists[iu] / 2.0
    
    # Combine all distances
    all_dists = np.concatenate([boundary, pair_dists])
    
    # Smooth min approximation: min(x) ≈ -1/b * log(sum(exp(-b*x)))
    # To avoid overflow, shift by max
    neg_dists = -all_dists
    max_neg = np.max(neg_dists)
    if not np.isfinite(max_neg):
        return 1e10
    log_sum = max_neg + np.log(np.sum(np.exp(neg_dists - max_neg)))
    smooth_min = -log_sum / b
    
    # We want to maximize smooth_min, so minimize -smooth_min
    return -smooth_min


def _local_optimize(points, maxiter=200, b=50.0):
    """
    Refine points using L-BFGS-B with the smooth objective.
    """
    x0 = points.flatten()
    
    def objective(x):
        return _smooth_objective(x, b=b)
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', 
                       options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 50})
        cand = res.x.reshape(-1, 2)
        # Ensure validity: project back if outside
        if np.all(np.isfinite(cand)):
            norms = np.linalg.norm(cand, axis=1)
            for i in range(len(cand)):
                if norms[i] > 1.0:
                    cand[i] = cand[i] / norms[i] * 0.9999
            # Verify improvement
            cand_score = _score_vec(cand)
            orig_score = _score_vec(points)
            if cand_score >= orig_score - 1e-10:
                return cand
    except Exception:
        pass
    return points.copy()


def _anneal(points, start_time, deadline, T_start=0.02, T_min=0.001, cooling=0.999):
    """
    Simulated annealing to escape local minima.
    """
    pts = points.copy()
    best_pts = pts.copy()
    best_score = _score_vec(best_pts)
    current_score = best_score
    
    T = T_start
    step = 0
    
    while time.time() < deadline - 2.0:
        step += 1
        
        # Generate perturbation
        n_move = np.random.randint(1, 6)
        idx = np.random.choice(N, n_move, replace=False)
        
        # Scale perturbation by temperature
        delta = np.random.normal(0, T * 0.1, size=(n_move, 2))
        
        new_pts = pts.copy()
        new_pts[idx] += delta
        
        new_score = _score_vec(new_pts)
        
        # Metropolis criterion
        if new_score >= current_score:
            pts = new_pts
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
        else:
            prob = np.exp((new_score - current_score) / T)
            if np.random.random() < prob:
                pts = new_pts
                current_score = new_score
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
            
        # Periodic local optimization
        if step % 200 == 0 and time.time() < deadline - 10.0:
            opt_pts = _local_optimize(pts, maxiter=50, b=100.0)
            opt_score = _score_vec(opt_pts)
            if opt_score > current_score:
                pts = opt_pts
                current_score = opt_score
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts.copy()
            # Reheat slightly
            T = max(T, 0.005)

    return best_pts, best_score


def _ring_parametric_objective(params, ring_config):
    """
    Objective function for ring-based parametric optimization.
    params: [r1, r2, ..., rk, theta1_1, theta1_2, ..., theta1_n1, theta2_1, ...]
    where rk are ring radii and theta_i_j are angular offsets for ring i.
    """
    # Parse params
    n_rings = len(ring_config)
    ring_radii = params[:n_rings]
    angle_params = params[n_rings:]
    
    # Build points
    points = []
    angle_idx = 0
    for ring_idx, count in enumerate(ring_config):
        if count == 1 and ring_idx == 0:
            # Center point
            points.append([0.0, 0.0])
        else:
            r = ring_radii[ring_idx]
            for k in range(count):
                base_angle = 2 * np.pi * k / count
                offset = angle_params[angle_idx]
                angle_idx += 1
                angle = base_angle + offset
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                points.append([x, y])
    
    points = np.array(points)
    if len(points) != N:
        return 1e10
    
    # Compute score (we want to maximize, so return negative)
    score = _score_vec(points)
    return -score


def _optimize_ring_params(ring_config, start_time, deadline):
    """
    Optimize ring radii and angular offsets using L-BFGS-B.
    """
    n_rings = len(ring_config)
    # Initial guess for ring radii
    initial_radii = np.array([(i + 1) * 0.18 for i in range(n_rings)])
    # Initial guess for angular offsets (all zero)
    n_angles = sum(count for count in ring_config if count > 1)
    initial_angles = np.zeros(n_angles)
    
    x0 = np.concatenate([initial_radii, initial_angles])
    
    def objective(x):
        return _ring_parametric_objective(x, ring_config)
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B',
                       options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100})
        
        # Reconstruct points from optimized params
        params = res.x
        ring_radii = params[:n_rings]
        angle_params = params[n_rings:]
        
        points = []
        angle_idx = 0
        for ring_idx, count in enumerate(ring_config):
            if count == 1 and ring_idx == 0:
                points.append([0.0, 0.0])
            else:
                r = ring_radii[ring_idx]
                for k in range(count):
                    base_angle = 2 * np.pi * k / count
                    offset = angle_params[angle_idx]
                    angle_idx += 1
                    angle = base_angle + offset
                    x = r * np.cos(angle)
                    y = r * np.sin(angle)
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) == N and np.all(np.isfinite(points)):
            # Project back if outside
            norms = np.linalg.norm(points, axis=1)
            for i in range(len(points)):
                if norms[i] > 1.0:
                    points[i] = points[i] / norms[i] * 0.9999
            
            return points
    except Exception:
        pass
    
    return None


def _binary_search_polish(points, start_time, deadline, tol=1e-10, max_iter=100):
    """
    Binary search polish: find the maximum radius r such that all circles
    fit without overlap. This is a high-precision final tuning step.
    """
    pts = points.copy()
    best_r = _score_vec(pts)
    
    # Binary search on radius
    r_low = best_r * 0.99
    r_high = best_r * 1.01
    
    for _ in range(max_iter):
        if time.time() > deadline - 1.0:
            break
        if r_high - r_low < tol:
            break
        
        r_mid = (r_low + r_high) / 2
        
        # Try to fit circles of radius r_mid
        # Scale points to see if they fit
        # Actually, we need to check if current points work for radius r_mid
        # This is tricky - let's use a different approach
        
        # Instead, let's do a local optimization targeting radius r_mid
        # For now, just track the best score from local optimization
        opt_pts = _local_optimize(pts, maxiter=100, b=200.0)
        opt_score = _score_vec(opt_pts)
        
        if opt_score > best_r:
            best_r = opt_score
            pts = opt_pts
            r_low = opt_score
        else:
            r_high = opt_score
    
    return pts, best_r


def construct():
    start_time = time.time()
    
    # 1. Ring-based initialization
    points = _ring_init(N)
    best_points = points.copy()
    best_score = _score_vec(best_points)
    
    # 2. Ring parametric optimization for different configurations
    ring_configs = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 20, 11],
        [1, 6, 14, 18, 11],
    ]
    
    for config in ring_configs:
        if time.time() > start_time + DEADLINE - 20:
            break
        if sum(config) != N:
            continue
        
        ring_pts = _optimize_ring_params(config, start_time, start_time + DEADLINE - 10)
        if ring_pts is not None:
            ring_score = _score_vec(ring_pts)
            if ring_score > best_score:
                best_score = ring_score
                best_points = ring_pts.copy()
    
    # 3. Initial local optimization
    if time.time() < start_time + DEADLINE - 15:
        opt_points = _local_optimize(best_points, maxiter=200, b=50.0)
        opt_score = _score_vec(opt_points)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_points.copy()
    
    # 4. Simulated Annealing
    if time.time() < start_time + DEADLINE - 10:
        sa_points, sa_score = _anneal(best_points, start_time, start_time + DEADLINE - 5)
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points.copy()
    
    # 5. Binary search polish for high precision
    if time.time() < start_time + DEADLINE - 3:
        polished_pts, polished_score = _binary_search_polish(
            best_points, start_time, start_time + DEADLINE - 1, 
            tol=1e-10, max_iter=50)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_pts.copy()
    
    # 6. Final high-precision polish
    if time.time() < start_time + DEADLINE - 1:
        final_pts = _local_optimize(best_points, maxiter=500, b=200.0)
        final_score = _score_vec(final_pts)
        if final_score > best_score:
            best_points = final_pts
            best_score = final_score
    
    return best_points
