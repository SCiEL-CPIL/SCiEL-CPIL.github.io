"""Packing of 50 equal circles in a unit square via Soft-Min L-BFGS with adaptive beta."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
HALF = 0.5

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
        
        coeff = 0.5 * wm
        
        np.add.at(grad, im, coeff[:, None] * unit)
        np.add.at(grad, jm, -coeff[:, None] * unit)
    
    return grad

def _lbfgs_optimize(pts0, beta, deadline, maxiter=300):
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
                       options={'maxiter': maxiter, 'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 20})
        pts = res.x.reshape(N, 2)
        pts = np.clip(pts, -HALF, HALF)
        return pts
    except Exception:
        return pts0

def _hex_init(rng, jitter=0.0):
    """Hexagonal close-packed lattice initialization."""
    pts = []
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

def _grid_init(rng, jitter=0.0):
    """Grid initialization with 7x7 + extras."""
    pts = []
    # Try a 7x7 grid which gives 49 points, then add 1 more
    n = 7
    for i in range(n):
        for j in range(n):
            x = -HALF + (i + 0.5) / n
            y = -HALF + (j + 0.5) / n
            pts.append([x, y])
    # Add one more point
    pts.append([0.0, 0.0])
    pts = np.array(pts[:N])
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
    return np.clip(pts, -HALF, HALF)

def _random_init(rng):
    """Random initialization."""
    pts = rng.uniform(-HALF + 0.05, HALF - 0.05, size=(N, 2))
    return pts

def construct():
    start = time.time()
    deadline = start + 140.0  # Safety margin (150s limit)
    rng = np.random.default_rng(42)
    
    # Multiple initializations to explore different basins
    inits = [
        _hex_init(rng, jitter=0.005),
        _hex_init(rng, jitter=0.02),
        _grid_init(rng, jitter=0.005),
        _random_init(rng),
        _hex_init(rng, jitter=0.05),
    ]
    
    best_pts = None
    best_r = 0.0
    
    # Progressive Beta Schedule - start low, increase to sharpen
    betas = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    
    # For each initialization, run the progressive beta schedule
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
        
        current_pts = init_pts.copy()
        init_r = _compute_r(current_pts)
        
        if init_r > best_r:
            best_r = init_r
            best_pts = current_pts.copy()
        
        for beta in betas:
            if time.time() > deadline - 5.0:
                break
            
            # Optimize with current beta
            new_pts = _lbfgs_optimize(current_pts, beta, deadline - 2.0)
            new_r = _compute_r(new_pts)
            
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts.copy()
                current_pts = new_pts.copy()
            else:
                # Try a small perturbation to escape local minima
                kick_size = 0.005 * (1.0 + init_idx * 0.5)
                kick = rng.normal(0, kick_size, size=(N, 2))
                kicked_pts = np.clip(current_pts + kick, -HALF, HALF)
                
                kicked_opt_pts = _lbfgs_optimize(kicked_pts, beta, deadline - 2.0)
                kicked_r = _compute_r(kicked_opt_pts)
                
                if kicked_r > best_r:
                    best_r = kicked_r
                    best_pts = kicked_opt_pts.copy()
                    current_pts = kicked_opt_pts.copy()
    
    # Final high-precision polish with very high beta
    if time.time() < deadline - 5.0 and best_pts is not None:
        final_betas = [10000, 20000, 50000]
        for beta in final_betas:
            if time.time() > deadline - 1.0:
                break
            new_pts = _lbfgs_optimize(best_pts, beta, deadline - 0.5, maxiter=200)
            new_r = _compute_r(new_pts)
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts.copy()
    
    # One more polish pass with the highest beta
    if time.time() < deadline - 0.5 and best_pts is not None:
        new_pts = _lbfgs_optimize(best_pts, 100000, deadline - 0.1, maxiter=100)
        new_r = _compute_r(new_pts)
        if new_r > best_r:
            best_r = new_r
            best_pts = new_pts.copy()
    
    return best_pts
