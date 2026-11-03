"""Combined program: ring parameterization + energy-based gradient optimization."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _compute_score(P):
    """Calculate the radius score for a given set of points."""
    if P.shape != (N, 2) or not np.all(np.isfinite(P)):
        return -np.inf
    
    d_bound = 1.0 - np.linalg.norm(P, axis=1)
    r_bound = np.min(d_bound)
    
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    r_pair = min_pair_dist / 2.0
    
    return min(r_pair, r_bound)

def _energy(P, r):
    """Smooth energy: sum of squared overlaps."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    overlaps_cc = np.maximum(2.0 * r - dist, 0.0)
    
    dist_origin = np.linalg.norm(P, axis=1)
    overlaps_cb = np.maximum(dist_origin + r - 1.0, 0.0)
    
    return 0.5 * (np.sum(overlaps_cc**2) + np.sum(overlaps_cb**2))

def _energy_grad(P, r):
    """Gradient of the energy function."""
    grad = np.zeros_like(P)
    
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
    
    dist_origin = np.linalg.norm(P, axis=1)
    V_cb = np.maximum(r + dist_origin - 1.0, 0.0)
    
    inv_dist_origin = np.zeros_like(dist_origin)
    np.divide(1.0, dist_origin, out=inv_dist_origin, where=dist_origin > 0)
    
    grad += (V_cb * inv_dist_origin)[:, np.newaxis] * P
    
    return grad

def _get_ring_positions(params, ring_counts):
    """Generate points from ring parameters: [R1..Rk, off1..offk]."""
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

def _optimize_positions_energy(P_init, r_target, deadline):
    """Minimize energy for fixed r using L-BFGS-B with gradient."""
    start = time.time()
    
    def objective(P_flat):
        P = P_flat.reshape(N, 2)
        return _energy(P, r_target)
    
    def grad(P_flat):
        P = P_flat.reshape(N, 2)
        return _energy_grad(P, r_target).flatten()
        
    try:
        res = minimize(objective, P_init.flatten(), jac=grad, method='L-BFGS-B', 
                       options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
        P_opt = res.x.reshape(N, 2)
        return P_opt
    except Exception:
        return P_init

def _optimize_positions_direct(P_init, deadline):
    """Direct optimization of positions to maximize score using smooth surrogate."""
    start = time.time()
    
    def neg_score(P_flat):
        P = P_flat.reshape(N, 2)
        score = _compute_score(P)
        if not np.isfinite(score):
            return 1e10
        # Add penalty for points outside
        norms = np.linalg.norm(P, axis=1)
        penalty = np.sum(np.maximum(norms - 1.0, 0.0)**2) * 100
        return -score + penalty
    
    try:
        res = minimize(neg_score, P_init.flatten(), method='Nelder-Mead',
                       options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-12,
                                'adaptive': True})
        P_opt = res.x.reshape(N, 2)
        return P_opt
    except Exception:
        return P_init

def _force_directed_refine(P, r, deadline, max_iters=3000):
    """Force-directed refinement."""
    P_curr = P.copy()
    r_curr = r
    alpha = 0.005
    
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        forces = np.zeros_like(P_curr)
        
        diff = P_curr[:, np.newaxis, :] - P_curr[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        target = 2.0 * r_curr
        mask = dist < target * 1.5
        
        if np.any(mask):
            overlap = np.zeros_like(dist)
            overlap[mask] = target - dist[mask]
            overlap[overlap < 0] = 0
            
            inv_dist = np.zeros_like(dist)
            np.divide(1.0, dist, out=inv_dist, where=dist > 0)
            
            weight = overlap * inv_dist
            weight[~mask] = 0
            
            sum_w = np.sum(weight, axis=1)
            term1 = P_curr * sum_w[:, np.newaxis]
            term2 = (P_curr.T @ weight).T
            
            forces += term1 - term2
        
        dist_origin = np.linalg.norm(P_curr, axis=1)
        limit = 1.0 - r_curr
        outside = dist_origin > limit
        if np.any(outside):
            viol = dist_origin[outside] - limit
            unit_vec = P_curr[outside] / dist_origin[outside, np.newaxis]
            forces[outside] -= unit_vec * viol * 1.0
        
        forces -= 0.01 * P_curr
        
        P_new = P_curr + alpha * forces
        new_score = _compute_score(P_new)
        
        if new_score > r_curr:
            P_curr = P_new
            r_curr = new_score
            alpha = min(alpha * 1.1, 0.05)
        else:
            alpha *= 0.95
        
        if alpha < 1e-7:
            break
    
    return P_curr, r_curr

def _get_hex_initial():
    """Hexagonal lattice initialization."""
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

def _search_ring_params(ring_counts, deadline):
    """Differential evolution over ring radii and offsets."""
    k = len(ring_counts)
    bounds = [(0.0, 1.0)] * k + [(0.0, 2 * np.pi)] * k
    
    def neg_score(params):
        P = _get_ring_positions(params, ring_counts)
        score = _compute_score(P)
        return -score if np.isfinite(score) else 1e10
        
    start = time.time()
    try:
        max_iter = 40 if time.time() < deadline - 30 else 20
        res = differential_evolution(neg_score, bounds, maxiter=max_iter, popsize=12, 
                                     tol=1e-5, seed=42, polish=True, 
                                     workers=1, init='latinhypercube')
        best_params = res.x
        best_P = _get_ring_positions(best_params, ring_counts)
        return best_P
    except Exception:
        return _get_hex_initial()

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Try multiple ring configurations
    ring_configs = [
        [1, 12, 18, 19],   # center + 3 rings
        [1, 10, 15, 24],   # center + 3 rings
        [1, 14, 16, 19],   # center + 3 rings
        [12, 18, 20],      # 3 rings, no center
        [1, 12, 18, 19],   # duplicate for robustness
    ]
    
    P_best = _get_hex_initial()
    score_best = _compute_score(P_best)
    
    # Stage 1: Global search over ring parameters
    for ring_counts in ring_configs:
        if time.time() > deadline - 40.0:
            break
        P_ring = _search_ring_params(ring_counts, deadline)
        score_ring = _compute_score(P_ring)
        if score_ring > score_best:
            P_best = P_ring
            score_best = score_ring
    
    # Stage 2: Energy-based refinement with increasing r
    if time.time() < deadline - 25.0:
        r_target = score_best
        for _ in range(3):
            P_opt = _optimize_positions_energy(P_best, r_target, deadline)
            score_opt = _compute_score(P_opt)
            if score_opt > score_best:
                P_best = P_opt
                score_best = score_opt
                r_target = score_best
            else:
                break
    
    # Stage 3: Direct Nelder-Mead refinement
    if time.time() < deadline - 15.0:
        P_nm = _optimize_positions_direct(P_best, deadline)
        score_nm = _compute_score(P_nm)
        if score_nm > score_best:
            P_best = P_nm
            score_best = score_nm
    
    # Stage 4: Force-directed refinement
    if time.time() < deadline - 10.0:
        P_fd, r_fd = _force_directed_refine(P_best, score_best, deadline)
        if r_fd > score_best:
            P_best = P_fd
            score_best = r_fd
    
    # Stage 5: Final high-precision polish
    if time.time() < deadline - 5.0:
        def final_neg_score(P_flat):
            P = P_flat.reshape(N, 2)
            score = _compute_score(P)
            if not np.isfinite(score):
                return 1e10
            norms = np.linalg.norm(P, axis=1)
            penalty = np.sum(np.maximum(norms - 1.0, 0.0)**2) * 100
            return -score + penalty
        
        try:
            res = minimize(final_neg_score, P_best.flatten(), method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-13,
                                    'adaptive': True})
            P_polish = res.x.reshape(N, 2)
            score_polish = _compute_score(P_polish)
            if score_polish > score_best:
                P_best = P_polish
                score_best = score_polish
        except Exception:
            pass
    
    return P_best
# EVOLVE-BLOCK-END
