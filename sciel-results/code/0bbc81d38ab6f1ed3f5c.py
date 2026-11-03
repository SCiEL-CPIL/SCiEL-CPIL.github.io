"""Pack 50 equal circles in the unit disk via ring-based parametric optimization + gradient polish + binary search radius."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
DEADLINE = 140.0  # safety margin before 150s limit


def _compute_score(pts):
    """Compute the packing radius: min of half-min-pair-dist and min-container-margin."""
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return -np.inf
    # Container constraint
    container_margin = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    min_container = np.min(container_margin)
    
    # Pairwise distances (use squared for speed, only sqrt the min)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_sq = np.min(dist_sq)
    min_pair = np.sqrt(min_pair_sq) / 2.0
    
    return min(min_container, min_pair)


def _compute_score_fast(pts, moved_idx=None):
    """Faster score computation when only one point moved. Returns (score, min_pair, min_container)."""
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return -np.inf, np.inf, np.inf
    
    # Container constraint
    container_margin = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    min_container = np.min(container_margin)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_sq = np.min(dist_sq)
    min_pair = np.sqrt(min_pair_sq) / 2.0
    
    return min(min_container, min_pair), min_pair, min_container


def _energy(P, r, beta=20.0):
    """
    Smooth energy function using soft-min approximation.
    beta controls smoothness (higher = sharper).
    """
    # Circle-circle overlaps
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Soft max for circle-circle overlaps: smooth approximation of max(0, 2r - d)
    raw_cc = 2.0 * r - dist
    scale = beta
    soft_cc = (1.0 / scale) * np.logaddexp(0, scale * raw_cc)
    
    # Circle-boundary overlaps
    dist_origin = np.linalg.norm(P, axis=1)
    raw_cb = dist_origin + r - 1.0
    soft_cb = (1.0 / scale) * np.logaddexp(0, scale * raw_cb)
    
    return 0.5 * (np.sum(soft_cc**2) + np.sum(soft_cb**2))


def _energy_grad(P, r, beta=20.0):
    """
    Gradient of the smooth energy function.
    """
    grad = np.zeros_like(P)
    scale = beta
    
    # Pairwise
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    raw_cc = 2.0 * r - dist
    sig_cc = 1.0 / (1.0 + np.exp(-scale * raw_cc))
    sig_cc = np.clip(sig_cc, 1e-12, 1.0 - 1e-12)
    
    inv_dist = np.zeros_like(dist)
    np.divide(1.0, dist, out=inv_dist, where=dist > 0)
    
    soft_cc_val = (1.0 / scale) * np.logaddexp(0, scale * raw_cc)
    
    W = soft_cc_val * sig_cc * inv_dist
    np.fill_diagonal(W, 0.0)
    
    sum_W = np.sum(W, axis=1)
    term1 = (P.T @ W).T
    term2 = P * sum_W[:, np.newaxis]
    
    grad += term1 - term2
    
    # Boundary
    dist_origin = np.linalg.norm(P, axis=1)
    raw_cb = dist_origin + r - 1.0
    sig_cb = 1.0 / (1.0 + np.exp(-scale * raw_cb))
    sig_cb = np.clip(sig_cb, 1e-12, 1.0 - 1e-12)
    
    soft_cb_val = (1.0 / scale) * np.logaddexp(0, scale * raw_cb)
    
    inv_dist_origin = np.zeros_like(dist_origin)
    np.divide(1.0, dist_origin, out=inv_dist_origin, where=dist_origin > 0)
    
    grad += (soft_cb_val * sig_cb * inv_dist_origin)[:, np.newaxis] * P
    
    return grad


def _get_ring_initialization(params, ring_counts):
    """
    Generate points based on concentric rings.
    params: [R1, R2, ..., Rk, offset1, offset2, ..., offsetk]
    ring_counts: list of number of points in each ring.
    """
    P = np.zeros((N, 2))
    k = len(ring_counts)
    radii = params[:k]
    offsets = params[k:2*k]
    
    count = 0
    for i, n_pts in enumerate(ring_counts):
        if n_pts == 1:
            P[count] = [0.0, 0.0]
            count += 1
        else:
            R = radii[i]
            offset = offsets[i]
            angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False) + offset
            P[count:count+n_pts, 0] = R * np.cos(angles)
            P[count:count+n_pts, 1] = R * np.sin(angles)
            count += n_pts
    
    return P


def _optimize_positions(P_init, r_target, maxiter=300, beta=20.0):
    """
    Optimize positions P to minimize smooth energy for a fixed r_target.
    Returns (P_opt, min_energy).
    """
    def objective(P_flat):
        P = P_flat.reshape(N, 2)
        return _energy(P, r_target, beta=beta)
    
    def grad(P_flat):
        P = P_flat.reshape(N, 2)
        return _energy_grad(P, r_target, beta=beta).flatten()
        
    try:
        res = minimize(objective, P_init.flatten(), jac=grad, method='L-BFGS-B', 
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
        P_opt = res.x.reshape(N, 2)
        min_energy = objective(P_opt)
        return P_opt, min_energy
    except Exception:
        return P_init, objective(P_init.flatten())


def _parametric_score(params, ring_counts):
    """Evaluate the parametric ring configuration."""
    P = _get_ring_initialization(params, ring_counts)
    return _compute_score(P)


def _optimize_parametric_ring(ring_counts, seed=42, maxiter=60, deadline=None):
    """
    Optimize ring radii and offsets using differential evolution.
    """
    k = len(ring_counts)
    # Bounds: radii in [0, 0.95], offsets in [0, 2*pi]
    bounds = [(0.0, 0.95)] * k + [(0.0, 2 * np.pi)] * k
    
    def neg_score(params):
        if deadline is not None and time.time() > deadline:
            return 0.0  # Infeasible
        s = _parametric_score(params, ring_counts)
        return -s
        
    try:
        res = differential_evolution(neg_score, bounds, maxiter=maxiter, popsize=15, 
                                     tol=1e-6, seed=seed, polish=False, 
                                     workers=1, init='latinhypercube')
        
        best_params = res.x
        best_P = _get_ring_initialization(best_params, ring_counts)
        return best_P, -res.fun
        
    except Exception:
        # Fallback: try a few random configurations
        rng = np.random.default_rng(seed)
        best_P = None
        best_score = -np.inf
        for _ in range(20):
            params = np.array([rng.uniform(0.1, 0.9) for _ in range(k)] + 
                             [rng.uniform(0, 2*np.pi) for _ in range(k)])
            P = _get_ring_initialization(params, ring_counts)
            s = _compute_score(P)
            if s > best_score:
                best_score = s
                best_P = P
        if best_P is None:
            best_P = _get_hex_initial_points()
            best_score = _compute_score(best_P)
        return best_P, best_score


def _get_hex_initial_points():
    """Fallback initialization using hexagonal lattice."""
    spacing = 0.22
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        margin = spacing / 2.0 
        mask = radius <= 1.0 - margin
        points, radius = points[mask], radius[mask]
        if len(points) >= N:
            indices = np.argsort(radius, kind="stable")
            return points[indices[:N]]
        spacing *= 0.95
        if spacing < 0.1:
            break
    
    # Last resort: grid
    points = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = i * 0.2
            y = j * 0.2
            if x*x + y*y <= 0.85:
                points.append([x, y])
    points = np.array(points[:N])
    return points


def _local_search(pts, n_iters=2000, seed=12345, deadline=None):
    """
    Local search: perturb one point at a time, accept if score improves.
    Uses adaptive step size and temperature annealing.
    """
    rng = np.random.default_rng(seed)
    best_pts = pts.copy()
    best_score = _compute_score(best_pts)
    
    step_size = 0.01
    temperature = 0.003
    no_improve_count = 0
    max_no_improve = 100
    best_step = step_size
    
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        idx = rng.integers(0, N)
        old_pos = best_pts[idx].copy()
        new_pos = old_pos + step_size * rng.normal(size=2)
        
        # Keep inside container
        new_r = np.sqrt(np.sum(new_pos ** 2))
        if new_r > 0.995:
            new_pos *= 0.995 / new_r
        
        test_pts = best_pts.copy()
        test_pts[idx] = new_pos
        new_score, _, _ = _compute_score_fast(test_pts, moved_idx=idx)
        
        delta = new_score - best_score
        if delta > 0 or (temperature > 1e-8 and rng.random() < np.exp(delta / temperature)):
            best_pts[idx] = new_pos
            if new_score > best_score:
                best_score = new_score
                no_improve_count = 0
                best_step = step_size
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            # Reset and adjust
            temperature = max(temperature * 0.5, 1e-8)
            step_size = min(step_size * 1.3, 0.04)
            no_improve_count = 0
        
        # Gentle annealing
        temperature *= 0.9998
        step_size = max(0.001, step_size * 0.99995)
    
    return best_pts, best_score


def _binary_search_radius(P_init, r_low, r_high, max_iterations=12, deadline=None):
    """
    Binary search on radius using smooth optimization to find best feasible packing.
    Returns (best_P, best_score, best_r).
    """
    best_P = P_init.copy()
    best_score = _compute_score(P_init)
    best_r = r_low
    
    for _ in range(max_iterations):
        if deadline is not None and time.time() > deadline - 10.0:
            break
        if r_high - r_low < 1e-9:
            break
            
        r_mid = (r_low + r_high) / 2.0
        P_init_cur = best_P.copy()
        
        # Try to optimize for r_mid
        P_opt, E = _optimize_positions(P_init_cur, r_mid, maxiter=200, beta=20.0)
        score_opt = _compute_score(P_opt)
        
        # Check if feasible: energy near zero and score at least r_mid
        if E < 1e-5 and score_opt >= r_mid * 0.999:
            if score_opt > best_score:
                best_P = P_opt
                best_score = score_opt
                best_r = r_mid
            r_low = r_mid
        else:
            r_high = r_mid
    
    return best_P, best_score, best_r


def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # 5s buffer before 120s limit
    
    # Define ring structures to try
    ring_structures = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 14, 17],
        [1, 6, 12, 16, 15],
        [1, 6, 12, 15, 16],
        [1, 6, 10, 14, 19],
        [1, 6, 12, 17, 14],
        [1, 6, 11, 13, 19],
    ]
    
    # Start with hexagonal initialization
    P_best = _get_hex_initial_points()
    score_best = _compute_score(P_best)
    
    # Try each ring structure with parametric optimization
    for ring_counts in ring_structures:
        if time.time() > deadline - 40.0:
            break
        
        # 1. Global parametric search for ring radii and offsets
        P_candidate, score_candidate = _optimize_parametric_ring(
            ring_counts, seed=42, maxiter=50, deadline=deadline - 20.0)
        
        if score_candidate > score_best:
            P_best = P_candidate
            score_best = score_candidate
        
        # 2. Binary search on radius to push further
        if time.time() > deadline - 25.0:
            break
            
        r_low = max(score_best * 0.995, 0.12)
        r_high = 0.145
        
        P_opt, score_opt, _ = _binary_search_radius(
            P_candidate, r_low, r_high, max_iterations=10, deadline=deadline)
        
        if score_opt > score_best:
            P_best = P_opt
            score_best = score_opt
    
    # Final local search refinement - use remaining time
    if time.time() < deadline - 15.0:
        remaining_time = deadline - 15.0 - time.time()
        remaining_iters = min(8000, int(remaining_time * 800))
        remaining_iters = max(1000, remaining_iters)
        P_best, score_best = _local_search(P_best, n_iters=remaining_iters, seed=99999, deadline=deadline - 15.0)
    
    # Final L-BFGS-B polish with progressively higher beta
    if time.time() < deadline - 8.0:
        for beta_pass in [30.0, 60.0, 120.0, 250.0]:
            if time.time() > deadline - 3.0:
                break
            r_target = score_best * 1.00001
            P_opt, E = _optimize_positions(P_best, r_target, maxiter=600, beta=beta_pass)
            polished_score = _compute_score(P_opt)
            if polished_score > score_best:
                P_best = P_opt
                score_best = polished_score
        
        # Ultra-tight polish with multiple passes
        if time.time() < deadline - 2.0:
            for _ in range(3):
                if time.time() > deadline - 1.0:
                    break
                r_target = score_best * 1.000002
                P_opt, E = _optimize_positions(P_best, r_target, maxiter=800, beta=500.0)
                polished_score = _compute_score(P_opt)
                if polished_score > score_best:
                    P_best = P_opt
                    score_best = polished_score
                else:
                    break
    
    # Validate and ensure all points are strictly inside
    radii = np.sqrt(np.sum(P_best ** 2, axis=1))
    if np.any(~np.isfinite(P_best)) or np.any(radii >= 1.0):
        P_best = _get_hex_initial_points()
    else:
        # Push any points that are too close to the boundary inward
        for i in range(N):
            r = radii[i]
            if r > 0.999:
                P_best[i] *= 0.999 / r
    
    return P_best
