"""Packing 50 equal circles in a unit circle using smooth energy minimization with ring structures."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
DEADLINE = 135.0  # safety margin before 150s kill


def _compute_score(P):
    """Calculate the radius score for a given set of points."""
    P = np.asarray(P)
    if P.shape != (N, 2) or not np.all(np.isfinite(P)):
        return 0.0
    
    # Distance to boundary
    d_bound = 1.0 - np.linalg.norm(P, axis=1)
    r_bound = np.min(d_bound)
    
    # Pairwise distances
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    r_pair = min_pair_dist / 2.0
    
    return max(0.0, min(r_pair, r_bound))


def _energy(P, r):
    """
    Smooth energy function: sum of squared overlaps.
    This is differentiable, allowing gradient-based optimization.
    """
    # Circle-circle overlaps
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    overlaps_cc = np.maximum(2.0 * r - dist, 0.0)
    
    # Circle-boundary overlaps
    dist_origin = np.linalg.norm(P, axis=1)
    overlaps_cb = np.maximum(dist_origin + r - 1.0, 0.0)
    
    return 0.5 * (np.sum(overlaps_cc**2) + np.sum(overlaps_cb**2))


def _energy_grad(P, r):
    """
    Gradient of the energy function.
    """
    grad = np.zeros_like(P)
    
    # Pairwise
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    V_cc = np.maximum(2.0 * r - dist, 0.0)
    inv_dist = np.zeros_like(dist)
    np.divide(1.0, dist, out=inv_dist, where=dist > 0)
    
    W = V_cc * inv_dist
    
    sum_W = np.sum(W, axis=1) 
    term1 = (P.T @ W).T 
    term2 = P * sum_W[:, np.newaxis] 
    
    grad += term1 - term2
    
    # Boundary
    dist_origin = np.linalg.norm(P, axis=1)
    V_cb = np.maximum(r + dist_origin - 1.0, 0.0)
    
    inv_dist_origin = np.zeros_like(dist_origin)
    np.divide(1.0, dist_origin, out=inv_dist_origin, where=dist_origin > 0)
    
    grad += (V_cb * inv_dist_origin)[:, np.newaxis] * P
    
    return grad


def _get_ring_initialization(params, ring_counts):
    """
    Generate points based on concentric rings.
    params: [R1, R2, ..., Rk, offset1, offset2, ..., offsetk]
    ring_counts: list of number of points in each ring.
    """
    P = np.zeros((N, 2))
    k = len(ring_counts)
    
    # Radii are the first k params
    radii = params[:k]
    # Offsets are the next k params
    offsets = params[k:2*k]
    
    count = 0
    for i, n_pts in enumerate(ring_counts):
        if n_pts == 1:
            # Center point
            P[count] = [0.0, 0.0]
            count += 1
        else:
            R = max(0.0, radii[i])
            offset = offsets[i]
            angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False) + offset
            P[count:count+n_pts, 0] = R * np.cos(angles)
            P[count:count+n_pts, 1] = R * np.sin(angles)
            count += n_pts
            
    return P


def _optimize_positions(P_init, r_target, time_limit=5.0):
    """
    Optimize positions P to minimize energy for a fixed r_target.
    Returns (P_opt, min_energy).
    """
    start_time = time.time()
    
    def objective(P_flat):
        P = P_flat.reshape(N, 2)
        return _energy(P, r_target)
    
    def grad(P_flat):
        P = P_flat.reshape(N, 2)
        return _energy_grad(P, r_target).flatten()
        
    try:
        res = minimize(objective, P_init.flatten(), jac=grad, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        P_opt = res.x.reshape(N, 2)
        min_energy = objective(P_opt)
        return P_opt, min_energy
    except Exception:
        return P_init, objective(P_init.flatten())


def _get_hex_initial_points():
    """Fallback initialization using hexagonal lattice."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        margin = spacing / 2.0 
        points, radius = points[radius <= 1.0 - margin], radius[radius <= 1.0 - margin]
        if len(points) >= N:
            indices = np.argsort(radius, kind="stable")
            return points[indices[:N]]
        spacing *= 0.98


def _global_search_ring_params(ring_counts, time_limit=30.0):
    """
    Use differential evolution to find good ring radii and offsets.
    """
    k = len(ring_counts)
    # Bounds for radii: 0 to 1
    # Bounds for offsets: 0 to 2*pi
    bounds = [(0.0, 1.0)] * k + [(0.0, 2 * np.pi)] * k
    
    def neg_score(params):
        P = _get_ring_initialization(params, ring_counts)
        # Quick score check
        score = _compute_score(P)
        return -score
        
    start_time = time.time()
    
    try:
        # Use differential evolution for global search
        res = differential_evolution(neg_score, bounds, maxiter=50, popsize=15, 
                                     tol=1e-5, seed=42, polish=True, 
                                     workers=1,
                                     init='latinhypercube')
        
        best_params = res.x
        best_P = _get_ring_initialization(best_params, ring_counts)
        
    except Exception:
        # Fallback to heuristic if DE fails or takes too long
        best_P = _get_hex_initial_points()
        
    return best_P


def _polish(P, r_target, time_limit=10.0):
    """
    Final high-precision polish pass with tighter tolerances.
    Uses the smooth energy formulation with very tight convergence criteria.
    """
    start_time = time.time()
    
    def objective(P_flat):
        P = P_flat.reshape(N, 2)
        return _energy(P, r_target)
    
    def grad(P_flat):
        P = P_flat.reshape(N, 2)
        return _energy_grad(P, r_target).flatten()
        
    try:
        res = minimize(objective, P.flatten(), jac=grad, method='L-BFGS-B', 
                       options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
        P_opt = res.x.reshape(N, 2)
        return P_opt
    except Exception:
        return P


def construct():
    t_start = time.time()
    
    # Define ring structures for N=50
    ring_structures = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 14, 17],
        [1, 6, 12, 16, 15],
        [1, 6, 12, 15, 16],
        [1, 6, 10, 14, 19],
        [1, 6, 10, 12, 21],
    ]
    
    # Start with hexagonal initialization
    P_best = _get_hex_initial_points()
    score_best = _compute_score(P_best)
    
    # For each ring structure, optimize and refine
    for counts in ring_structures:
        if time.time() - t_start > DEADLINE - 20:
            break
        
        # 1. Global search for ring parameters
        P_candidate = _global_search_ring_params(counts, time_limit=min(20.0, DEADLINE - (time.time() - t_start) - 25))
        score_candidate = _compute_score(P_candidate)
        
        if score_candidate > score_best:
            P_best = P_candidate.copy()
            score_best = score_candidate
        
        # 2. Binary search on r with energy minimization
        r_low = score_best * 0.95
        r_high = 0.14
        
        # Ensure r_low is feasible
        P_opt, E = _optimize_positions(P_candidate, r_low, time_limit=5.0)
        if E < 1e-6:
            score = _compute_score(P_opt)
            if score > score_best:
                P_best = P_opt.copy()
                score_best = score
            r_low = score
        
        # Binary search for maximum feasible r
        for _ in range(12):
            if time.time() - t_start > DEADLINE - 20:
                break
            r_mid = 0.5 * (r_low + r_high)
            P_opt, E = _optimize_positions(P_candidate, r_mid, time_limit=5.0)
            if E < 1e-6:
                score = _compute_score(P_opt)
                if score > score_best:
                    P_best = P_opt.copy()
                    score_best = score
                r_low = r_mid
            else:
                r_high = r_mid
    
    # Final polish pass with tighter tolerances
    if time.time() - t_start < DEADLINE - 10:
        r_target = score_best * 1.001
        P_polished = _polish(P_best, r_target, time_limit=15.0)
        score_polished = _compute_score(P_polished)
        if score_polished > score_best:
            P_best = P_polished
            score_best = score_polished
    
    return P_best
