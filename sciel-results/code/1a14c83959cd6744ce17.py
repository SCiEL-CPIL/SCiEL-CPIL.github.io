"""Packing of 50 equal circles in a unit square via Soft-Min L-BFGS."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
HALF = 0.5
# Best known radius for reference (to cap binary search or target)
R_KNOWN = 0.071377103865

def _compute_r(pts):
    """Compute the feasible radius for a configuration (double precision)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # boundary: min distance from each point to the square edge
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    r = float(np.min(d))
    # pairwise
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r = min(r, float(np.min(dist)) / 2.0)
    return r

def _soft_min_radius(pts, beta):
    """
    Compute the soft-minimum radius surrogate.
    r_soft = max_d - (1/beta) * log(sum(exp(-beta * (d_k - max_d))))
    """
    # Wall distances
    d_left = pts[:, 0] + HALF
    d_right = HALF - pts[:, 0]
    d_bottom = pts[:, 1] + HALF
    d_top = HALF - pts[:, 1]
    
    # Pairwise distances (divided by 2 because radius is half distance)
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    iu = np.triu_indices(N, k=1)
    pair_dists = dist[iu] / 2.0
    
    all_d = np.concatenate([d_left, d_right, d_bottom, d_top, pair_dists])
    
    max_d = np.max(all_d)
    # Stable log-sum-exp
    exp_terms = np.exp(-beta * (all_d - max_d))
    log_sum = np.log(np.sum(exp_terms))
    r_soft = max_d - log_sum / beta
    
    return float(r_soft)

def _soft_min_grad(pts, beta):
    """
    Compute the gradient of the soft-minimum radius surrogate.
    grad = sum_k w_k * grad(d_k)
    """
    d_left = pts[:, 0] + HALF
    d_right = HALF - pts[:, 0]
    d_bottom = pts[:, 1] + HALF
    d_top = HALF - pts[:, 1]
    
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    iu = np.triu_indices(N, k=1)
    pair_dists = dist[iu] / 2.0
    
    all_d = np.concatenate([d_left, d_right, d_bottom, d_top, pair_dists])
    
    max_d = np.max(all_d)
    exp_terms = np.exp(-beta * (all_d - max_d))
    weights = exp_terms / np.sum(exp_terms)
    
    grad = np.zeros((N, 2))
    
    # Wall gradients
    # d_left[i] = x_i + HALF -> grad_x = 1
    # d_right[i] = HALF - x_i -> grad_x = -1
    # d_bottom[i] = y_i + HALF -> grad_y = 1
    # d_top[i] = HALF - y_i -> grad_y = -1
    
    w_left = weights[:N]
    w_right = weights[N:2*N]
    w_bottom = weights[2*N:3*N]
    w_top = weights[3*N:4*N]
    
    grad[:, 0] += w_left
    grad[:, 0] -= w_right
    grad[:, 1] += w_bottom
    grad[:, 1] -= w_top
    
    # Pairwise gradients
    i_idx = iu[0]
    j_idx = iu[1]
    w_pair = weights[4*N:]
    
    # Only process pairs with significant weight to save time
    mask = w_pair > 1e-12
    if np.any(mask):
        im = i_idx[mask]
        jm = j_idx[mask]
        wm = w_pair[mask]
        
        dij = pts[im] - pts[jm]  # (M, 2)
        dists = np.linalg.norm(dij, axis=1)  # (M,)
        safe_dists = np.where(dists > 1e-15, dists, 1.0)
        unit = dij / safe_dists[:, None]  # (M, 2)
        
        # d(pair_dist/2)/d(pts[i]) = 0.5 * (pts[i]-pts[j])/||pts[i]-pts[j]|| = 0.5 * unit
        # d(pair_dist/2)/d(pts[j]) = -0.5 * unit
        # Wait, pair_dists = dist / 2.
        # grad of (dist/2) wrt pts[i] is 0.5 * unit.
        # So coeff is 0.5 * wm.
        
        coeff = 0.5 * wm
        
        np.add.at(grad, im, coeff[:, None] * unit)
        np.add.at(grad, jm, -coeff[:, None] * unit)
    
    return grad

def _lbfgs_optimize(pts0, beta, deadline, maxiter=500):
    """
    Maximize soft-min radius using L-BFGS-B.
    """
    def obj(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        return -_soft_min_radius(p, beta)
    
    def grad(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        return -_soft_min_grad(p, beta).ravel()
    
    x0 = pts0.ravel()
    try:
        res = minimize(obj, x0, method='L-BFGS-B',
                       jac=grad,
                       bounds=[(-HALF, HALF)] * (2 * N),
                       options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-12})
        pts = res.x.reshape(N, 2)
        pts = np.clip(pts, -HALF, HALF)
        return pts
    except Exception:
        return pts0

def _hex_init(rng, jitter=0.0):
    """Hexagonal close-packed lattice initialization."""
    pts = []
    # Approximate spacing for 50 circles in unit square
    # Area per circle ~ 1/50 = 0.02. Side ~ 0.14.
    # Let's try a=0.13
    a = 0.13
    dx = a
    dy = a * np.sqrt(3) / 2
    row = 0
    while len(pts) < N:
        offset = (dx / 2) if (row % 2) else 0.0
        x = -HALF + a + offset
        while len(pts) < N and x < HALF - a + 1e-9:
            pts.append([x, -HALF + a + row * dy])
            x += dx
        row += 1
    pts = np.array(pts[:N])
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
    return np.clip(pts, -HALF, HALF)

def construct():
    start = time.time()
    deadline = start + 145.0  # Safety margin
    rng = np.random.default_rng(42)
    
    # Initial guess
    pts = _hex_init(rng, jitter=0.01)
    best_pts = pts.copy()
    best_r = _compute_r(pts)
    
    # Progressive Beta Schedule
    # Start low to find a good basin, increase to sharpen the minimum
    betas = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    
    current_pts = pts.copy()
    
    for beta in betas:
        if time.time() > deadline - 5.0:
            break
            
        # Optimize
        new_pts = _lbfgs_optimize(current_pts, beta, deadline - 2.0)
        
        # Check improvement
        new_r = _compute_r(new_pts)
        
        if new_r > best_r:
            best_r = new_r
            best_pts = new_pts.copy()
            current_pts = new_pts.copy()
        else:
            # If no improvement, try a small kick to escape local minima
            # This is more effective than restarting from scratch
            kick = rng.normal(0, 0.01, size=(N, 2))
            kicked_pts = np.clip(current_pts + kick, -HALF, HALF)
            
            # Try optimizing from kicked position
            kicked_opt_pts = _lbfgs_optimize(kicked_pts, beta, deadline - 2.0)
            kicked_r = _compute_r(kicked_opt_pts)
            
            if kicked_r > best_r:
                best_r = kicked_r
                best_pts = kicked_opt_pts.copy()
                current_pts = kicked_opt_pts.copy()
            else:
                # Keep current best if kick didn't help
                pass
                
    # Final high-precision polish with very high beta
    if time.time() < deadline - 5.0:
        final_betas = [10000, 50000]
        for beta in final_betas:
            if time.time() > deadline - 1.0:
                break
            new_pts = _lbfgs_optimize(best_pts, beta, deadline - 0.5, maxiter=100)
            new_r = _compute_r(new_pts)
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts.copy()

    return best_pts
