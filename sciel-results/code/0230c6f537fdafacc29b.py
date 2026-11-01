"""Packing 50 equal circles in a unit disk using contact-aware soft-max optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
START_TIME = None
DEADLINE = 140.0  # Safety margin before 150s kill


def _score(points):
    """Compute packing radius: min(min pairwise dist/2, min (1-|c_i|))."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    dists = np.linalg.norm(points, axis=1)
    if np.any(dists >= 1.0):
        return 0.0
    boundary_margin = np.min(1.0 - dists)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_mat, np.inf)
    min_pair_dist = np.min(dist_mat)
    return min(min_pair_dist / 2.0, boundary_margin)


def _ring_init(ring_sizes, ring_radii, angle_offsets=None):
    """Generate points from concentric rings."""
    points = np.zeros((N, 2))
    idx = 0
    if angle_offsets is None:
        angle_offsets = [0.0] * len(ring_sizes)
    for ring_idx, (size, radius) in enumerate(zip(ring_sizes, ring_radii)):
        offset = angle_offsets[ring_idx] if ring_idx < len(angle_offsets) else 0.0
        for j in range(size):
            angle = 2 * np.pi * j / size + offset
            points[idx] = [radius * np.cos(angle), radius * np.sin(angle)]
            idx += 1
    return points


def _hex_lattice_init(r_guess):
    """Hexagonal lattice initialization."""
    spacing = 2.0 * r_guess
    pts = []
    k = int(1.0 / spacing) + 3
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= (1.0 - r_guess) ** 2:
                pts.append([x, y])
    pts = np.array(pts)
    if len(pts) >= N:
        radii = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(radii, kind='stable')[:N]
        return pts[idx]
    else:
        return _ring_init([1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.75, 0.95])


def _random_init(n):
    """Random initialization inside unit circle."""
    angles = np.random.uniform(0, 2 * np.pi, n)
    radii = np.sqrt(np.random.uniform(0, 1, n))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _soft_max_objective(pts, r_target, beta):
    """
    Soft-max objective: approximates max of all constraint violations.
    Violations: boundary (|c_i| - (1-r)) and pairwise (2r - |c_i-c_j|).
    Returns (objective_value, gradient).
    """
    n = len(pts)
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    cont_viol = norms - (1.0 - r_target)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_viol = 2.0 * r_target - dists
    iu = np.triu_indices(n, k=1)
    pair_viol_flat = pair_viol[iu]
    
    all_viol = np.concatenate([cont_viol, pair_viol_flat])
    
    # Soft-max approximation of max
    max_v = np.max(all_viol)
    if max_v > 50.0 / beta:
        return max_v, np.zeros_like(pts)
    
    exp_v = np.exp(beta * (all_viol - max_v))
    sum_exp = np.sum(exp_v)
    soft_max_val = max_v + np.log(sum_exp) / beta
    weights = exp_v / sum_exp
    
    grad = np.zeros_like(pts)
    
    # Container gradient: d/dx_i (|x_i|) = x_i / |x_i|
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    unit_vecs = pts / safe_norms[:, np.newaxis]
    grad += weights[:n, np.newaxis] * unit_vecs
    
    # Pairwise gradient
    iw = weights[n:]
    valid = dists[iu] > 1e-10
    if np.any(valid):
        iw_valid = iw[valid]
        iu_valid = (iu[0][valid], iu[1][valid])
        d_valid = dists[iu_valid]
        diff_valid = pts[iu_valid[0]] - pts[iu_valid[1]]
        unit_diff = diff_valid / d_valid[:, np.newaxis]
        
        np.add.at(grad, iu_valid[0], -iw_valid[:, np.newaxis] * unit_diff)
        np.add.at(grad, iu_valid[1], iw_valid[:, np.newaxis] * unit_diff)
    
    return soft_max_val, grad


def _soft_max_optimize(pts_init, r_target, start_beta=50.0, end_beta=10000.0,
                       n_stages=6, max_iter_per_stage=500, tol=1e-12):
    """
    Optimize using soft-max objective with beta annealing.
    Returns optimized positions.
    """
    current_pts = pts_init.copy()
    
    for stage in range(n_stages):
        if time.time() - START_TIME > DEADLINE - 5:
            break
        
        frac = stage / max(n_stages - 1, 1)
        beta = start_beta * (end_beta / start_beta) ** frac
        
        # Define objective and gradient for L-BFGS-B
        def obj_grad(x):
            pts = x.reshape(N, 2)
            val, grad = _soft_max_objective(pts, r_target, beta)
            return val, grad.reshape(-1)
        
        x0 = current_pts.reshape(-1)
        try:
            result = minimize(
                obj_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': max_iter_per_stage, 'ftol': tol, 'gtol': 1e-10, 'maxls': 50}
            )
            current_pts = result.x.reshape(N, 2)
        except Exception:
            pass
    
    return current_pts


def _check_feasible(pts, r_target, tol=1e-6):
    """Check if configuration is feasible for given radius."""
    val, _ = _soft_max_objective(pts, r_target, beta=10000.0)
    return val < tol


def construct():
    """Construct the circle packing."""
    global START_TIME
    START_TIME = time.time()
    
    best_positions = None
    best_score = 0.0
    
    # Multiple diverse initializations
    inits = [
        _hex_lattice_init(0.12),
        _ring_init([1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.75, 0.95]),
        _ring_init([1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.75, 0.95], 
                   angle_offsets=[0.0, np.pi/6, np.pi/12, np.pi/8, np.pi/10]),
        _random_init(N),
        _random_init(N),
    ]
    
    for init_idx, init_pos in enumerate(inits):
        if time.time() - START_TIME > DEADLINE - 10:
            break
        
        # Get initial score
        init_score = _score(init_pos)
        
        # Bisection search for optimal radius using soft-max optimizer
        lo, hi = 0.05, 0.15
        best_r_for_init = init_score
        best_pos_for_init = init_pos.copy()
        
        for bisect_iter in range(12):
            if time.time() - START_TIME > DEADLINE - 5:
                break
            
            mid_r = (lo + hi) / 2.0
            
            # Try to optimize for this radius
            optimized_pos = _soft_max_optimize(
                init_pos, mid_r,
                start_beta=50.0,
                end_beta=10000.0,
                n_stages=5,
                max_iter_per_stage=400
            )
            
            # Check feasibility
            if _check_feasible(optimized_pos, mid_r, tol=1e-6):
                actual_score = _score(optimized_pos)
                if actual_score > best_score:
                    best_score = actual_score
                    best_positions = optimized_pos.copy()
                if actual_score > best_r_for_init:
                    best_r_for_init = actual_score
                    best_pos_for_init = optimized_pos.copy()
                lo = mid_r
            else:
                hi = mid_r
        
        # Refine around best radius found for this initialization
        if best_r_for_init > 0:
            for refine_iter in range(4):
                if time.time() - START_TIME > DEADLINE - 3:
                    break
                test_r = best_r_for_init * 1.008
                optimized_pos = _soft_max_optimize(
                    best_pos_for_init, test_r,
                    start_beta=100.0,
                    end_beta=20000.0,
                    n_stages=4,
                    max_iter_per_stage=300
                )
                if _check_feasible(optimized_pos, test_r, tol=1e-7):
                    actual_score = _score(optimized_pos)
                    if actual_score > best_score:
                        best_score = actual_score
                        best_positions = optimized_pos.copy()
                    best_r_for_init = actual_score
                    best_pos_for_init = optimized_pos.copy()
                else:
                    break
    
    # Final polish with very high beta
    if best_positions is not None and best_score > 0:
        current_r = best_score
        for polish_iter in range(6):
            if time.time() - START_TIME > DEADLINE - 2:
                break
            test_r = current_r * 1.003
            optimized_pos = _soft_max_optimize(
                best_positions, test_r,
                start_beta=500.0,
                end_beta=50000.0,
                n_stages=3,
                max_iter_per_stage=200
            )
            if _check_feasible(optimized_pos, test_r, tol=1e-8):
                actual_score = _score(optimized_pos)
                if actual_score > best_score:
                    best_score = actual_score
                    best_positions = optimized_pos.copy()
                    current_r = actual_score
                else:
                    break
            else:
                break
    
    if best_positions is not None:
        return best_positions
    else:
        return _hex_lattice_init(0.12)
