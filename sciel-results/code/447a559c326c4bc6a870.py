"""Packing 200 equal circles in a unit square: L-BFGS-B optimization from diverse starts."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200
DEADLINE = 145.0  # seconds; leave margin before 150s kill

def _score(pts):
    """Return the common radius r for a valid configuration, or 0.0 if invalid."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Distance to boundary
    ax = np.abs(pts[:, 0])
    ay = np.abs(pts[:, 1])
    boundary = np.minimum(0.5 - ax, 0.5 - ay)
    min_boundary = np.min(boundary)
    if min_boundary < 0:
        return 0.0
    # Pairwise distances
    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    r = min(min_boundary, min_dist / 2.0)
    return r if r > 0 else 0.0

def _hex_init():
    """Generate a hexagonal lattice initialization."""
    # We want to fit N points in a hexagonal pattern.
    # Spacing s determines how many fit.
    # Area of hexagon packing is slightly denser than square.
    # Let's try a few spacings.
    best_pts = None
    best_r = 0
    # Estimate spacing: area per point ~ 1/N. Side ~ 1/sqrt(N).
    # Hex spacing s is roughly 1/sqrt(N) * 0.9 for density.
    s_est = 1.0 / np.sqrt(N) * 1.2 
    for s in np.linspace(s_est * 0.9, s_est * 1.1, 10):
        pts_list = []
        row = 0
        # y range [-0.5 + s/2, 0.5 - s/2]
        while True:
            y = -0.5 + s/2 + row * s * np.sqrt(3) / 2
            if y > 0.5 - s/2:
                break
            offset = (row % 2) * s / 2
            # x range [-0.5 + s/2 + offset, 0.5 - s/2]
            x = -0.5 + s/2 + offset
            while x <= 0.5 - s/2:
                pts_list.append((x, y))
                x += s
            row += 1
        if len(pts_list) < N:
            continue
        pts = np.array(pts_list[:N])
        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    if best_pts is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        pts = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = pts[:N].copy()
    return best_pts

def _grid_init():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    pts = np.array([(x, y) for y in ticks for x in ticks])
    return pts[:N].copy()

def _random_init(seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 2))

def _soft_min_neg(x, alpha):
    """
    Objective function for L-BFGS-B.
    x: flat array of size 2N.
    alpha: sharpness of the soft-min.
    Returns: -soft_min(distances)
    """
    pts = x.reshape(N, 2)
    
    # Distances to boundary
    # 0.5 - |x| and 0.5 - |y|
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    boundary_dists = np.minimum(d_x, d_y).flatten()
    
    # Pairwise distances
    # Only compute upper triangle to save memory/time?
    # cdist is fine for N=200.
    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    # We only need the finite distances
    pairwise_dists = dist[~np.isinf(dist)]
    
    # Combine all distances
    all_dists = np.concatenate([boundary_dists, pairwise_dists])
    
    # Soft min: -1/alpha * log(sum(exp(-alpha * d)))
    # To avoid overflow, shift by max
    m = np.max(all_dists)
    shifted = all_dists - m
    log_sum = m + np.log(np.sum(np.exp(-alpha * shifted)))
    soft_min_val = log_sum / alpha  # This is the soft max of -d, so soft min of d is -this?
    # Wait.
    # min(d_i) ≈ - (1/alpha) log(sum(exp(-alpha d_i)))
    # Let's verify. If d_i = 0 for one, and large for others.
    # exp(-alpha * 0) = 1. Others ~ 0.
    # log(1) = 0. Result 0. Correct.
    # If d_i = 1, 2. alpha=10.
    # exp(-10) + exp(-20) ~ 4.5e-5.
    # log(4.5e-5) ~ -9.98.
    # -1/10 * -9.98 = 0.998. Close to 1. Correct.
    
    val = - (1.0 / alpha) * np.log(np.sum(np.exp(-alpha * all_dists)))
    
    # We want to maximize min distance, so minimize -min distance.
    # The expression above is the soft MIN.
    # So we return - (soft min) to minimize it.
    return -val

def _optimize(pts, time_budget, alpha_start=10.0, alpha_end=100.0, max_iters=1000):
    """
    Run L-BFGS-B optimization on the initial points.
    """
    t0 = time.time()
    x0 = pts.flatten()
    
    best_x = x0.copy()
    best_val = _soft_min_neg(x0, alpha_start) # Minimize this
    
    # We can try increasing alpha over a few runs
    # Or just use a fixed high alpha.
    # Let's do a sequence of minimizations with increasing alpha.
    
    # Phase 1: Coarse optimization with lower alpha
    # Phase 2: Fine optimization with higher alpha
    
    phases = [
        {"alpha": 10.0, "max_iter": 200, "gtol": 1e-4},
        {"alpha": 50.0, "max_iter": 200, "gtol": 1e-5},
        {"alpha": 100.0, "max_iter": 200, "gtol": 1e-6},
    ]
    
    for phase in phases:
        if time.time() - t0 > time_budget:
            break
            
        alpha = phase["alpha"]
        
        # Wrap the objective with the specific alpha
        def obj(x):
            return _soft_min_neg(x, alpha)
        
        try:
            res = minimize(obj, x0, method='L-BFGS-B', 
                           bounds=[(-0.5, 0.5)] * (2 * N),
                           options={'maxiter': phase["max_iter"], 'gtol': phase["gtol"]})
            x_opt = res.x
            val_opt = obj(x_opt)
            
            # Check if it improved the best score found so far in this function
            # Note: val is -soft_min, so lower is better.
            # But we track best_x based on the actual score at the end?
            # Actually, soft_min with high alpha is a good proxy for min.
            # Let's just keep the result of the last phase as the best candidate
            # and verify with exact score.
            x0 = x_opt
        except Exception:
            pass
            
    return x0.reshape(N, 2)

def construct():
    t_start = time.time()
    
    initializers = [
        ("hex", _hex_init),
        ("grid", _grid_init),
        ("rand0", lambda: _random_init(0)),
        ("rand1", lambda: _random_init(42)),
    ]
    
    best_pts = None
    best_r = 0.0
    
    for name, init_fn in initializers:
        if time.time() > DEADLINE - 10:
            break
            
        pts = init_fn()
        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
            
        # Optimize
        remaining = DEADLINE - time.time()
        if remaining < 5:
            break
            
        # Allocate time proportional to remaining budget
        # Give more time to hex as it's usually better
        if name == "hex":
            time_budget = min(remaining * 0.4, 40.0)
        else:
            time_budget = min(remaining * 0.2, 20.0)
            
        opt_pts = _optimize(pts, time_budget)
        
        # Clamp to square
        opt_pts = np.clip(opt_pts, -0.5, 0.5)
        
        opt_r = _score(opt_pts)
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()
            
    # Final polish: run a short optimization on the best result found
    remaining = DEADLINE - time.time()
    if remaining > 5 and best_pts is not None:
        # Use a high alpha for fine-tuning
        time_budget = min(remaining * 0.5, 10.0)
        opt_pts = _optimize(best_pts, time_budget, alpha_start=100.0, alpha_end=200.0, max_iters=200)
        opt_pts = np.clip(opt_pts, -0.5, 0.5)
        opt_r = _score(opt_pts)
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()
            
    if best_pts is None:
        best_pts = _hex_init()
        
    return best_pts
