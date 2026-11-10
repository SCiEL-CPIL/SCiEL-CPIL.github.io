# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize, minimize_scalar
import time


def _compute_score(points):
    """Compute R = (d_min/d_max)^2 for a set of points."""
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2


def _surrogate_objective(x, p=6):
    """
    Smooth surrogate: minimize log(S_neg) + log(S_pos) where
    S_neg = sum d_i^(-p), S_pos = sum d_i^p
    This is equivalent to minimizing S_neg * S_pos but more numerically stable.
    """
    points = x.reshape(-1, 2)
    d = pdist(points)
    d = np.maximum(d, 1e-8)
    
    log_S_neg = np.sum(np.log(d) * (-p))  # log(sum d^(-p)) approx via log-sum-exp
    # More accurate: use log-sum-exp trick
    neg_powers = -p * np.log(d)
    log_S_neg = np.log(np.sum(np.exp(neg_powers - np.max(neg_powers)))) + np.max(neg_powers)
    
    pos_powers = p * np.log(d)
    log_S_pos = np.log(np.sum(np.exp(pos_powers - np.max(pos_powers)))) + np.max(pos_powers)
    
    return log_S_neg + log_S_pos


def _local_refine_direct(points, max_iter=3000, tol=1e-10):
    """
    Refine points by directly optimizing a smooth approximation of the score.
    We maximize (d_min/d_max)^2 by minimizing -log(d_min/d_max) = log(d_max) - log(d_min).
    Using power means as smooth approximations.
    """
    n = len(points)
    x0 = points.flatten()
    
    def neg_score(x):
        pts = x.reshape(-1, 2)
        d = pdist(pts)
        d = np.maximum(d, 1e-12)
        # Smooth min and max using power means with large p
        p = 20
        d_min_smooth = np.exp(-np.log(np.mean(d ** (-p))) / p)
        d_max_smooth = np.exp(np.log(np.mean(d ** p)) / p)
        if d_max_smooth < 1e-15:
            return 0.0
        ratio = d_min_smooth / d_max_smooth
        if ratio <= 0:
            return 0.0
        return -np.log(ratio)
    
    res = minimize(neg_score, x0, method='Nelder-Mead', 
                   options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
    
    best_points = res.x.reshape(-1, 2)
    return best_points


def _local_refine_coordinate(points, max_iter=2000):
    """
    Coordinate descent: optimize each point's (x,y) one at a time.
    For each point, find the position that maximizes the score.
    """
    pts = points.copy()
    n = len(pts)
    
    for iteration in range(max_iter):
        improved = False
        for i in range(n):
            # Save current position
            orig = pts[i].copy()
            
            # Try to optimize point i by minimizing the negative score
            # with other points fixed
            best_pos = orig.copy()
            best_val = _compute_score(pts)
            
            # Use Nelder-Mead on just 2 variables (x, y) of point i
            def obj_2d(xy):
                pts[i] = xy
                return -_compute_score(pts)
            
            res = minimize(obj_2d, orig, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.fun < best_val - 1e-12:
                pts[i] = res.x
                best_val = res.fun
                improved = True
            else:
                pts[i] = orig
        
        if not improved:
            break
    
    return pts


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 570  # Safety margin
    
    best_points = None
    best_score = -1.0
    
    def update_best(pts):
        nonlocal best_points, best_score
        s = _compute_score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. 4x4 grid
    grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
    grid = (grid - grid.mean(axis=0)) / grid.std()
    initial_configs.append(grid.copy())
    
    # 2. Hexagonal lattice
    hex_pts = []
    for row in range(4):
        for col in range(4):
            x = col + (0.5 if row % 2 else 0)
            y = row * np.sqrt(3) / 2
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    hex_pts = (hex_pts - hex_pts.mean(axis=0)) / hex_pts.std()
    initial_configs.append(hex_pts.copy())
    
    # 3. Concentric rings (4 points per ring, 4 rings)
    ring_pts = []
    for ring in range(4):
        r = (ring + 1) * 0.5
        for k in range(4):
            angle = 2 * np.pi * k / 4 + ring * np.pi / 4
            ring_pts.append([r * np.cos(angle), r * np.sin(angle)])
    ring_pts = np.array(ring_pts)
    ring_pts = (ring_pts - ring_pts.mean(axis=0)) / ring_pts.std()
    initial_configs.append(ring_pts.copy())
    
    # 4. Random with minimum distance constraint
    rng = np.random.default_rng(12345)
    random_pts = rng.random((16, 2)) * 2 - 1
    # Simple jitter to avoid coincidences
    for _ in range(100):
        d = pdist(random_pts)
        if d.min() > 0.05:
            break
        random_pts += rng.random((16, 2)) * 0.01
    random_pts = (random_pts - random_pts.mean(axis=0)) / random_pts.std()
    initial_configs.append(random_pts.copy())
    
    # 5. Another random config
    rng2 = np.random.default_rng(99999)
    random_pts2 = rng2.random((16, 2)) * 2 - 1
    for _ in range(100):
        d = pdist(random_pts2)
        if d.min() > 0.05:
            break
        random_pts2 += rng2.random((16, 2)) * 0.01
    random_pts2 = (random_pts2 - random_pts2.mean(axis=0)) / random_pts2.std()
    initial_configs.append(random_pts2.copy())
    
    # Evaluate initial configs
    for config in initial_configs:
        update_best(config)
    
    # Strategy: For each initial config, run local refinement, then DE if time permits
    configs_to_refine = initial_configs[:]
    
    # First pass: quick local refinement on all initial configs
    for config in configs_to_refine:
        if time.time() > deadline - 30:
            break
        try:
            refined = _local_refine_direct(config, max_iter=1000)
            update_best(refined)
        except Exception:
            pass
    
    # Second pass: coordinate descent on best candidates
    if best_points is not None and time.time() < deadline - 60:
        try:
            coord_refined = _local_refine_coordinate(best_points, max_iter=500)
            update_best(coord_refined)
        except Exception:
            pass
    
    # Third pass: Differential Evolution from multiple seeds
    if time.time() < deadline - 120:
        bounds = [(-3, 3)] * (n * d)
        
        for seed_val in [42, 123, 777]:
            if time.time() > deadline - 90:
                break
            try:
                res = differential_evolution(
                    lambda x: _surrogate_objective(x, p=6),
                    bounds=bounds,
                    seed=seed_val,
                    maxiter=500,
                    popsize=12,
                    tol=1e-5,
                    polish=False
                )
                candidate = res.x.reshape(-1, 2)
                update_best(candidate)
                
                # Refine the DE result
                if time.time() < deadline - 60:
                    refined = _local_refine_direct(candidate, max_iter=1500)
                    update_best(refined)
                    
                    if time.time() < deadline - 40:
                        coord_refined = _local_refine_coordinate(refined, max_iter=300)
                        update_best(coord_refined)
            except Exception:
                pass
    
    # Fourth pass: More aggressive local optimization on the best
    if best_points is not None and time.time() < deadline - 30:
        try:
            final_refined = _local_refine_direct(best_points, max_iter=2000)
            update_best(final_refined)
        except Exception:
            pass
        
        if best_points is not None and time.time() < deadline - 15:
            try:
                final_coord = _local_refine_coordinate(best_points, max_iter=500)
                update_best(final_coord)
            except Exception:
                pass
    
    # Fallback
    if best_points is None:
        best_points = initial_configs[0]
    
    # Final sanity check
    if best_points.shape != (16, 2):
        best_points = initial_configs[0]
    
    return best_points


# EVOLVE-BLOCK-END
