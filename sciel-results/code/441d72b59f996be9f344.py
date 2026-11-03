"""Pack 50 equal circles in the unit disk via smooth log-sum-exp objective + ring-based DE + multi-stage annealing polish."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
DEADLINE = 145.0  # safety margin before 150s limit


def _compute_score(pts):
    """Compute the packing radius: min of half-min-pair-dist and min-container-margin."""
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return -np.inf
    # Container constraint
    container_margin = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    min_container = np.min(container_margin)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    return min(min_container, min_pair)


def _soft_min_surrogate(pts, beta=50.0):
    """
    Smooth surrogate for the packing radius using log-sum-exp.
    Returns a smooth approximation to min(min_container, min_pair/2).
    """
    # Container margins
    dist_origin = np.sqrt(np.sum(pts ** 2, axis=1))
    container_margin = 1.0 - dist_origin
    
    # Pairwise distances (half)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_margin = dist / 2.0
    
    # Flatten both into a single vector of margins
    all_margins = np.concatenate([container_margin, pair_margin.ravel()])
    all_margins = np.maximum(all_margins, -1.0)  # clip for numerical stability
    
    # Soft min via log-sum-exp: -log(sum(exp(-beta * x))) / beta
    # Use stable form: m = min(x), result = m - log(sum(exp(-beta*(x-m))))/beta
    m = np.min(all_margins)
    if m < -0.5:
        return m  # very bad, just return the hard min
    exp_terms = np.exp(-beta * (all_margins - m))
    soft_min = m - np.log(np.sum(exp_terms)) / beta
    
    return soft_min


def _soft_min_grad(pts, beta=50.0):
    """
    Gradient of the soft-min surrogate.
    """
    # Container margins
    dist_origin = np.sqrt(np.sum(pts ** 2, axis=1))
    container_margin = 1.0 - dist_origin
    
    # Pairwise distances (half)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_margin = dist / 2.0
    
    all_margins = np.concatenate([container_margin, pair_margin.ravel()])
    all_margins_clipped = np.maximum(all_margins, -1.0)
    
    m = np.min(all_margins_clipped)
    if m < -0.5:
        # Hard min regime: return gradient of the hard min
        # This is approximate; in practice the soft-min should keep us away from this
        return _hard_min_grad(pts)
    
    exp_terms = np.exp(-beta * (all_margins_clipped - m))
    weights = exp_terms / np.sum(exp_terms)
    
    # Gradient w.r.t. pts
    grad = np.zeros_like(pts)
    
    # Container part: d/dp_i (1 - |p_i|) = -p_i / |p_i|
    w_container = weights[:N]
    for i in range(N):
        if dist_origin[i] > 1e-10:
            grad[i] -= w_container[i] * beta * pts[i] / dist_origin[i]
    
    # Pair part: d/dp_i (|p_i - p_j|/2) for each pair
    # For pair (i,j) with i<j: margin = |p_i - p_j|/2
    # d/dp_i = (p_i - p_j) / |p_i - p_j|
    # d/dp_j = (p_j - p_i) / |p_i - p_j|
    w_pair = weights[N:]
    for i in range(N):
        for j in range(i + 1, N):
            idx = i * N + j  # flat index in pair_margin (upper triangle)
            # Actually pair_margin is full NxN with diagonal inf, so flat index is i*N+j
            w = w_pair[i * N + j]
            if dist[i, j] > 1e-10:
                direction = (pts[i] - pts[j]) / dist[i, j]
                grad[i] += w * beta * direction
                grad[j] -= w * beta * direction
    
    return grad


def _hard_min_grad(pts):
    """Approximate gradient of the hard min (for fallback)."""
    dist_origin = np.sqrt(np.sum(pts ** 2, axis=1))
    container_margin = 1.0 - dist_origin
    min_container_idx = np.argmin(container_margin)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    i, j = np.unravel_index(np.argmin(dist), dist.shape)
    
    if min_container_idx is not None and container_margin[min_container_idx] <= min_pair / 2:
        grad = np.zeros_like(pts)
        if dist_origin[min_container_idx] > 1e-10:
            grad[min_container_idx] = -pts[min_container_idx] / dist_origin[min_container_idx]
        return grad
    else:
        grad = np.zeros_like(pts)
        if dist[i, j] > 1e-10:
            direction = (pts[i] - pts[j]) / dist[i, j]
            grad[i] += direction
            grad[j] -= direction
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
            count += 1 if n_pts == 0 else n_pts
            
    return P


def _polish_smooth(P_init, beta_start=10.0, beta_end=200.0, n_stages=5, maxiter_per_stage=150, deadline=None):
    """
    Multi-stage annealing polish using the smooth log-sum-exp objective.
    Gradually increase beta to transition from global to local optimization.
    """
    P = P_init.copy()
    betas = np.geomspace(beta_start, beta_end, n_stages)
    
    for beta in betas:
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        def objective(P_flat):
            P = P_flat.reshape(N, 2)
            return -_soft_min_surrogate(P, beta)
        
        def grad(P_flat):
            P = P_flat.reshape(N, 2)
            return -_soft_min_grad(P, beta).flatten()
        
        try:
            res = minimize(objective, P.flatten(), jac=grad, method='L-BFGS-B',
                           options={'maxiter': maxiter_per_stage, 'ftol': 1e-14, 'gtol': 1e-12})
            P_new = res.x.reshape(N, 2)
            # Validate: ensure points are inside the disk
            radii = np.sqrt(np.sum(P_new ** 2, axis=1))
            if np.all(radii < 1.0) and np.all(np.isfinite(P_new)):
                P = P_new
            else:
                # Clip points that went outside
                for i in range(N):
                    if radii[i] >= 1.0:
                        P_new[i] *= 0.99 / radii[i]
                P = P_new
        except Exception:
            pass
    
    return P


def _polish_exact(P_init, r_target, maxiter=300, deadline=None):
    """
    Final exact polish: minimize the maximum overlap for a given r_target.
    Uses the energy function approach for feasibility checking.
    """
    def energy(P_flat):
        P = P_flat.reshape(N, 2)
        # Circle-circle overlaps
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        overlaps_cc = np.maximum(2.0 * r_target - dist, 0.0)
        
        # Circle-boundary overlaps
        dist_origin = np.linalg.norm(P, axis=1)
        overlaps_cb = np.maximum(dist_origin + r_target - 1.0, 0.0)
        
        return 0.5 * (np.sum(overlaps_cc**2) + np.sum(overlaps_cb**2))
    
    def energy_grad(P_flat):
        P = P_flat.reshape(N, 2)
        grad = np.zeros_like(P)
        
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        V_cc = np.maximum(2.0 * r_target - dist, 0.0)
        inv_dist = np.zeros_like(dist)
        np.divide(1.0, dist, out=inv_dist, where=dist > 0)
        
        W = V_cc * inv_dist
        sum_W = np.sum(W, axis=1)
        term1 = (P.T @ W).T
        term2 = P * sum_W[:, np.newaxis]
        grad += term1 - term2
        
        dist_origin = np.linalg.norm(P, axis=1)
        V_cb = np.maximum(r_target + dist_origin - 1.0, 0.0)
        inv_dist_origin = np.zeros_like(dist_origin)
        np.divide(1.0, dist_origin, out=inv_dist_origin, where=dist_origin > 0)
        grad += (V_cb * inv_dist_origin)[:, np.newaxis] * P
        
        return grad.flatten()
    
    try:
        res = minimize(energy, P_init.flatten(), jac=energy_grad, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-13})
        P_opt = res.x.reshape(N, 2)
        # Validate
        radii = np.sqrt(np.sum(P_opt ** 2, axis=1))
        if np.all(radii < 1.0) and np.all(np.isfinite(P_opt)):
            return P_opt
        else:
            return P_init
    except Exception:
        return P_init


def _local_search(pts, n_iters=2000, seed=12345, deadline=None):
    """
    Local search: perturb one point at a time, accept if score improves.
    """
    rng = np.random.default_rng(seed)
    best_pts = pts.copy()
    best_score = _compute_score(best_pts)
    
    step_size = 0.01
    temperature = 0.003
    no_improve_count = 0
    max_no_improve = 150
    
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        idx = rng.integers(0, N)
        old_pos = best_pts[idx].copy()
        new_pos = old_pos + step_size * rng.normal(size=2)
        
        # Keep inside container
        new_r = np.sqrt(np.sum(new_pos ** 2))
        if new_r > 0.99:
            new_pos *= 0.99 / new_r
        
        test_pts = best_pts.copy()
        test_pts[idx] = new_pos
        new_score = _compute_score(test_pts)
        
        delta = new_score - best_score
        if delta > 0 or (temperature > 1e-6 and rng.random() < np.exp(delta / temperature)):
            best_pts[idx] = new_pos
            if new_score > best_score:
                best_score = new_score
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            temperature = max(temperature * 1.5, 0.005)
            step_size = min(step_size * 1.3, 0.04)
            no_improve_count = 0
        
        temperature *= 0.9998
        step_size = max(0.002, step_size * 0.9998)
    
    return best_pts, best_score


def _global_search_ring_params(ring_counts, seed=42, maxiter=30, deadline=None):
    """
    Use differential evolution to find good ring radii and offsets.
    """
    k = len(ring_counts)
    bounds = [(0.0, 1.0)] * k + [(0.0, 2 * np.pi)] * k
    
    def neg_score(params):
        P = _get_ring_initialization(params, ring_counts)
        score = _compute_score(P)
        return -score
    
    try:
        res = differential_evolution(neg_score, bounds, maxiter=maxiter, popsize=15,
                                     tol=1e-6, seed=seed, polish=False,
                                     workers=1, init='latinhypercube')
        
        best_params = res.x
        best_P = _get_ring_initialization(best_params, ring_counts)
        
        # Quick smooth polish
        if deadline is not None and time.time() < deadline - 15.0:
            best_P = _polish_smooth(best_P, beta_start=10.0, beta_end=50.0, 
                                    n_stages=3, maxiter_per_stage=80, deadline=deadline)
        
    except Exception:
        best_P = _get_hex_initial_points()
        
    return best_P


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


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # 10s buffer before 150s limit
    
    # Define ring structures to try
    ring_structures = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 14, 17],
        [1, 6, 12, 16, 15],
        [1, 6, 12, 15, 16],
        [1, 6, 10, 14, 19],
        [1, 6, 12, 12, 19],
    ]
    
    # Start with hexagonal initialization
    P_best = _get_hex_initial_points()
    score_best = _compute_score(P_best)
    
    # Try each ring structure
    for ring_counts in ring_structures:
        if time.time() > deadline - 50.0:
            break
        
        # 1. Global search for ring parameters
        P_candidate = _global_search_ring_params(ring_counts, seed=42, maxiter=30, deadline=deadline)
        score_candidate = _compute_score(P_candidate)
        
        if score_candidate > score_best:
            P_best = P_candidate
            score_best = score_candidate
        
        # 2. Multi-stage smooth polish on the candidate
        if time.time() < deadline - 30.0:
            P_polished = _polish_smooth(P_candidate, beta_start=5.0, beta_end=100.0,
                                        n_stages=5, maxiter_per_stage=120, deadline=deadline)
            score_polished = _compute_score(P_polished)
            if score_polished > score_best:
                P_best = P_polished
                score_best = score_polished
        
        # 3. Binary search on radius with exact polish
        if time.time() < deadline - 20.0:
            r_low = score_best * 0.99
            r_high = 0.145
            
            for _ in range(6):
                if time.time() > deadline - 15.0:
                    break
                if r_high - r_low < 1e-8:
                    break
                    
                r_mid = (r_low + r_high) / 2.0
                P_init = P_best.copy()
                
                P_opt = _polish_exact(P_init, r_mid, maxiter=200, deadline=deadline)
                score_opt = _compute_score(P_opt)
                
                if score_opt >= r_mid * 0.999:
                    if score_opt > score_best:
                        P_best = P_opt
                        score_best = score_opt
                    r_low = r_mid
                else:
                    r_high = r_mid
    
    # Final local search refinement
    if time.time() < deadline - 10.0:
        remaining_iters = min(2500, int((deadline - 10.0 - time.time()) * 400))
        remaining_iters = max(300, remaining_iters)
        P_best, score_best = _local_search(P_best, n_iters=remaining_iters, seed=99999, deadline=deadline - 10.0)
    
    # Final high-precision polish pass
    if time.time() < deadline - 5.0:
        P_final = _polish_smooth(P_best, beta_start=50.0, beta_end=300.0,
                                 n_stages=3, maxiter_per_stage=200, deadline=deadline)
        score_final = _compute_score(P_final)
        if score_final > score_best:
            P_best = P_final
            score_best = score_final
        
        # One more exact polish if time allows
        if time.time() < deadline - 2.0:
            r_target = score_best * 1.00001
            P_opt2 = _polish_exact(P_best, r_target, maxiter=150, deadline=deadline)
            score_opt2 = _compute_score(P_opt2)
            if score_opt2 > score_best:
                P_best = P_opt2
                score_best = score_opt2
    
    # Validate
    radii = np.sqrt(np.sum(P_best ** 2, axis=1))
    if np.any(~np.isfinite(P_best)) or np.any(radii >= 1.0):
        P_best = _get_hex_initial_points()
    
    return P_best
