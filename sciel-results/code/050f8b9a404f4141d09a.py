"""
Pack 200 equal circles in a unit square using smooth soft-min optimization
combined with L-BFGS-B and high-precision polish for maximum radius.
"""
import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 200
_TIME_LIMIT = 140  # Leave margin for 150s kill

def _compute_radius_fast(pts):
    """
    Fast radius computation using cdist.
    """
    if len(pts) == 0 or not np.all(np.isfinite(pts)):
        return 0.0
    
    # Wall distances
    min_wall = np.min([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1])
    ])
    
    if min_wall < 0:
        return 0.0

    # Pairwise distances
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D) / 2.0
    
    return min(min_wall, min_pair)


def _soft_min_value_and_grad(pts, beta=50.0):
    """
    Compute soft-min of all pairwise distances (scaled) and wall distances,
    plus gradient w.r.t. points. Vectorized for speed.
    """
    n = len(pts)
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    D = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(D, np.inf)
    
    # Wall distances (4 per point)
    w_left = x + 0.5
    w_right = 0.5 - x
    w_bottom = y + 0.5
    w_top = 0.5 - y
    
    # Pair effective: D/2
    pair_eff = D / 2.0
    # Wall effective: the 4 wall distances
    wall_eff = np.column_stack([w_left, w_right, w_bottom, w_top])  # (n, 4)
    
    # Flatten all values, mask out diagonal
    pair_mask = ~np.eye(n, dtype=bool)
    all_vals = np.concatenate([pair_eff[pair_mask], wall_eff.ravel()])
    
    if len(all_vals) == 0:
        return 0.0, np.zeros_like(pts)
    
    # Soft-min: -logsumexp(-beta * vals) / beta
    shifted = -beta * all_vals
    max_shifted = np.max(shifted)
    logsumexp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    soft_min_val = -logsumexp / beta
    
    # Gradient w.r.t. all_vals: softmax weights
    weights = np.exp(shifted - max_shifted)
    weights /= np.sum(weights)
    
    # Build gradient
    grad = np.zeros_like(pts)
    
    # Pair part: reshape weights back to (n,n) with zero diagonal
    pair_weights = np.zeros((n, n))
    pair_weights[pair_mask] = weights[:pair_mask.sum()]
    safe_D = np.where(D > 1e-12, D, 1.0)
    factor = pair_weights / (2.0 * safe_D)
    factor = np.where(np.isfinite(factor), factor, 0.0)
    
    gx = np.sum(factor * dx, axis=1)
    gy = np.sum(factor * dy, axis=1)
    grad[:, 0] += gx
    grad[:, 1] += gy
    
    # Wall part
    wall_weights = weights[pair_mask.sum():]
    ww = wall_weights.reshape(n, 4)
    grad[:, 0] += ww[:, 0] - ww[:, 1]
    grad[:, 1] += ww[:, 2] - ww[:, 3]
    
    return soft_min_val, grad


def _project_into_square(pts):
    """Clip points to be inside the square."""
    return np.clip(pts, -0.5, 0.5)

def _local_optimize(coords, time_limit=None, start_time=None):
    """
    L-BFGS-B optimization using soft-min surrogate with increasing beta.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = _TIME_LIMIT - 10
    
    coords = coords.copy()
    n = len(coords)
    
    def make_objective(beta):
        def obj_and_grad(x):
            pts = x.reshape(-1, 2)
            val, grad = _soft_min_value_and_grad(pts, beta=beta)
            return -val, grad.ravel()
        return obj_and_grad
    
    best_r = _compute_radius_fast(coords)
    best_coords = coords.copy()
    
    # Multi-stage L-BFGS-B with increasing beta
    beta_schedule = [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    for beta in beta_schedule:
        if time.time() - start_time > time_limit:
            break
        
        obj_grad = make_objective(beta)
        x0 = best_coords.ravel()
        bounds = [(-0.5, 0.5)] * (2 * n)
        
        try:
            result = minimize(
                obj_grad[0],
                x0,
                jac=obj_grad[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8,
                         'maxls': 50, 'maxfun': 2000}
            )
            new_coords = result.x.reshape(-1, 2)
            new_coords = np.clip(new_coords, -0.5, 0.5)
            r_new = _compute_radius_fast(new_coords)
            
            if r_new > best_r + 1e-14:
                best_r = r_new
                best_coords = new_coords.copy()
        except Exception:
            pass
    
    return best_coords


def _high_beta_polish(coords, time_limit=15.0, start_time=None):
    """
    High-precision polish using very high beta soft-min with L-BFGS-B.
    """
    if start_time is None:
        start_time = time.time()
    
    coords = coords.copy()
    n = len(coords)
    
    best_r = _compute_radius_fast(coords)
    best_coords = coords.copy()
    
    beta_schedule = [2000.0, 5000.0, 10000.0, 20000.0]
    
    for beta in beta_schedule:
        if time.time() - start_time > time_limit:
            break
        
        def obj(x):
            pts = x.reshape(-1, 2)
            val, _ = _soft_min_value_and_grad(pts, beta=beta)
            return -val
        
        def grad(x):
            pts = x.reshape(-1, 2)
            _, g = _soft_min_value_and_grad(pts, beta=beta)
            return g.ravel()
        
        bounds = [(-0.5, 0.5)] * (2 * n)
        
        try:
            result = minimize(
                obj,
                best_coords.ravel(),
                jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10,
                         'maxls': 100, 'maxfun': 5000}
            )
            new_coords = result.x.reshape(-1, 2)
            new_coords = np.clip(new_coords, -0.5, 0.5)
            r_new = _compute_radius_fast(new_coords)
            
            if r_new > best_r + 1e-15:
                best_r = r_new
                best_coords = new_coords.copy()
        except Exception:
            pass
    
    return best_coords


def _direct_local_search(coords, time_limit=10.0, start_time=None):
    """
    Direct local search: iteratively push the closest pair apart and
    adjust wall-violating points. Works on the exact objective.
    """
    if start_time is None:
        start_time = time.time()
    
    coords = coords.copy()
    n = len(coords)
    best_r = _compute_radius_fast(coords)
    best_coords = coords.copy()
    
    max_iter = 5000
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Find current radius and what's limiting it
        D = cdist(coords, coords)
        np.fill_diagonal(D, np.inf)
        
        min_pair_dist = np.min(D)
        min_pair_idx = np.unravel_index(np.argmin(D), D.shape)
        
        min_wall = np.min([
            np.min(coords[:, 0]) + 0.5,
            0.5 - np.max(coords[:, 0]),
            np.min(coords[:, 1]) + 0.5,
            0.5 - np.max(coords[:, 1])
        ])
        
        r = min(min_pair_dist / 2.0, min_wall)
        
        if r > best_r + 1e-15:
            best_r = r
            best_coords = coords.copy()
        
        # Determine which constraint is active
        if min_pair_dist / 2.0 <= min_wall:
            # Pair is limiting: push them apart
            i, j = min_pair_idx
            diff = coords[i] - coords[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-12:
                direction = diff / dist
                # Push apart by a small amount
                push = 0.001 * (1.0 + it * 0.0001)
                coords[i] += direction * push
                coords[j] -= direction * push
        else:
            # Wall is limiting: push the closest point inward
            wl = np.min(coords[:, 0]) + 0.5
            wr = 0.5 - np.max(coords[:, 0])
            wb = np.min(coords[:, 1]) + 0.5
            wt = 0.5 - np.max(coords[:, 1])
            
            min_w = min(wl, wr, wb, wt)
            push = 0.001 * (1.0 + it * 0.0001)
            
            if min_w == wl:
                idx = np.argmin(coords[:, 0])
                coords[idx, 0] += push
            elif min_w == wr:
                idx = np.argmax(coords[:, 0])
                coords[idx, 0] -= push
            elif min_w == wb:
                idx = np.argmin(coords[:, 1])
                coords[idx, 1] += push
            else:
                idx = np.argmax(coords[:, 1])
                coords[idx, 1] -= push
        
        coords = np.clip(coords, -0.5, 0.5)
    
    return best_coords

def _hexagonal_initialization(n, jitter=0.0):
    """
    Generate a hexagonal layout with optional jitter.
    """
    m = int(np.ceil(np.sqrt(n)))
    r_est = 0.5 / m
    
    cols = int(1.0 / (2 * r_est)) + 2
    rows = int(1.0 / (r_est * np.sqrt(3))) + 2
    
    points = []
    for row in range(rows):
        y = -0.5 + r_est + row * r_est * np.sqrt(3)
        offset = 0.0 if row % 2 == 0 else r_est
        for col in range(cols):
            x = -0.5 + r_est + offset + col * 2 * r_est
            points.append([x, y])
            
    points = np.array(points)
    
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    if len(points) < n:
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        points = grid[:n]
    else:
        dists_to_center = np.sum(points**2, axis=1)
        indices = np.argsort(dists_to_center)
        points = points[indices[:n]]
    
    if jitter > 0:
        points += jitter * np.random.randn(n, 2)
        points = np.clip(points, -0.5, 0.5)
    
    return points

def _grid_initialization(n, jitter=0.0):
    """
    Generate a grid layout with optional jitter.
    """
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    
    if len(grid) < n:
        extra = n - len(grid)
        extra_pts = np.random.uniform(-0.45, 0.45, (extra, 2))
        grid = np.vstack([grid, extra_pts])
    
    grid = grid[:n]
    
    if jitter > 0:
        grid += jitter * np.random.randn(n, 2)
        grid = np.clip(grid, -0.5, 0.5)
    
    return grid

def _random_initialization(n):
    """
    Generate random points.
    """
    return np.random.uniform(-0.45, 0.45, (n, 2))

def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    
    initializers = [
        ("hex", lambda: _hexagonal_initialization(N, jitter=0.0)),
        ("hex_jitter", lambda: _hexagonal_initialization(N, jitter=0.01)),
        ("grid", lambda: _grid_initialization(N, jitter=0.0)),
        ("random", lambda: _random_initialization(N)),
    ]
    
    best_coords = None
    best_radius = 0.0
    
    # Phase 1: L-BFGS-B optimization on multiple starts
    for name, init_func in initializers:
        if time.time() > deadline - 50:
            break
        
        try:
            coords = init_func()
            r = _compute_radius_fast(coords)
            
            if r > best_radius:
                best_radius = r
                best_coords = coords.copy()
            
            remaining_time = deadline - time.time() - 40
            if remaining_time > 10:
                optimized = _local_optimize(coords,
                                           time_limit=remaining_time,
                                           start_time=start_time)
                r_opt = _compute_radius_fast(optimized)
                
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_coords = optimized.copy()
        except Exception:
            continue
    
    # Phase 2: High-beta L-BFGS-B polish
    if best_coords is not None:
        remaining_time = deadline - time.time() - 15
        if remaining_time > 5:
            try:
                polished = _high_beta_polish(best_coords,
                                             time_limit=remaining_time,
                                             start_time=start_time)
                r_pol = _compute_radius_fast(polished)
                if r_pol > best_radius:
                    best_radius = r_pol
                    best_coords = polished
            except Exception:
                pass
    
    # Phase 3: Direct local search on exact objective
    if best_coords is not None:
        remaining_time = deadline - time.time() - 3
        if remaining_time > 2:
            try:
                polished = _direct_local_search(best_coords,
                                                time_limit=remaining_time,
                                                start_time=start_time)
                r_pol = _compute_radius_fast(polished)
                if r_pol > best_radius:
                    best_radius = r_pol
                    best_coords = polished
            except Exception:
                pass
    
    # Final validation
    if best_coords is None or not np.all(np.isfinite(best_coords)):
        return _grid_initialization(N)
    
    best_coords = np.clip(best_coords, -0.4999, 0.4999)
    
    return best_coords

# EVOLVE-BLOCK-START
def compute_radius(coords):
    return _compute_radius_fast(coords)

def objective(coords):
    r = compute_radius(coords)
    return -r

def initial_hex_layout(n):
    return _hexagonal_initialization(n)

# EVOLVE-BLOCK-END
