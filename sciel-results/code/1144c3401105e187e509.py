"""Packing 200 equal circles in a unit square: multi-start L-BFGS-B with soft-min annealing and targeted local search."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

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
    """Generate a hexagonal lattice initialization with more spacing variants."""
    best_pts = None
    best_r = 0
    s_est = 1.0 / np.sqrt(N) * 1.2
    for s in np.linspace(s_est * 0.85, s_est * 1.15, 20):
        pts_list = []
        row = 0
        while True:
            y = -0.5 + s/2 + row * s * np.sqrt(3) / 2
            if y > 0.5 - s/2:
                break
            offset = (row % 2) * s / 2
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
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    boundary_dists = np.minimum(d_x, d_y).flatten()
    
    # Pairwise distances
    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    pairwise_dists = dist[~np.isinf(dist)]
    
    # Combine all distances
    all_dists = np.concatenate([boundary_dists, pairwise_dists])
    
    # Soft min: -1/alpha * log(sum(exp(-alpha * d)))
    val = - (1.0 / alpha) * np.log(np.sum(np.exp(-alpha * all_dists)))
    
    return -val

def _exact_min_neg(x):
    """
    Exact negative minimum distance objective for final polish.
    """
    pts = x.reshape(N, 2)
    
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    boundary_dists = np.minimum(d_x, d_y)
    
    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    pairwise_dists = dist[~np.isinf(dist)]
    
    min_boundary = np.min(boundary_dists)
    min_pairwise = np.min(pairwise_dists)
    
    r = min(min_boundary, min_pairwise / 2.0)
    return -r

def _targeted_local_search(pts, time_budget):
    """
    Targeted local search: identify the active constraints (closest pairs and boundary)
    and perform targeted moves to increase the minimum radius.
    """
    t0 = time.time()
    best_pts = pts.copy()
    best_r = _score(best_pts)
    
    rng = np.random.default_rng(999)
    
    for iteration in range(5000):
        if time.time() - t0 > time_budget:
            break
        
        pts = best_pts.copy()
        
        # Find the minimum radius and which constraints are active
        d_x = 0.5 - np.abs(pts[:, 0])
        d_y = 0.5 - np.abs(pts[:, 1])
        boundary_dists = np.minimum(d_x, d_y)
        
        dist = cdist(pts, pts)
        np.fill_diagonal(dist, np.inf)
        
        min_boundary_idx = np.argmin(boundary_dists)
        min_boundary_val = boundary_dists[min_boundary_idx]
        
        min_pair_idx = np.unravel_index(np.argmin(dist), dist.shape)
        min_pair_dist = dist[min_pair_idx]
        
        r = min(min_boundary_val, min_pair_dist / 2.0)
        
        # Try targeted moves
        improved = False
        
        if min_boundary_val <= min_pair_dist / 2.0:
            # Boundary constraint is active: push the closest circle inward
            i = min_boundary_idx
            if d_x[i] <= d_y[i]:
                # Closest to x-boundary
                if pts[i, 0] >= 0:
                    # Push left
                    for step in [0.001, 0.0005, 0.0001]:
                        trial = pts.copy()
                        trial[i, 0] -= step
                        trial = np.clip(trial, -0.5, 0.5)
                        trial_r = _score(trial)
                        if trial_r > best_r:
                            best_pts = trial
                            best_r = trial_r
                            improved = True
                            break
                else:
                    # Push right
                    for step in [0.001, 0.0005, 0.0001]:
                        trial = pts.copy()
                        trial[i, 0] += step
                        trial = np.clip(trial, -0.5, 0.5)
                        trial_r = _score(trial)
                        if trial_r > best_r:
                            best_pts = trial
                            best_r = trial_r
                            improved = True
                            break
            else:
                # Closest to y-boundary
                if pts[i, 1] >= 0:
                    # Push down
                    for step in [0.001, 0.0005, 0.0001]:
                        trial = pts.copy()
                        trial[i, 1] -= step
                        trial = np.clip(trial, -0.5, 0.5)
                        trial_r = _score(trial)
                        if trial_r > best_r:
                            best_pts = trial
                            best_r = trial_r
                            improved = True
                            break
                else:
                    # Push up
                    for step in [0.001, 0.0005, 0.0001]:
                        trial = pts.copy()
                        trial[i, 1] += step
                        trial = np.clip(trial, -0.5, 0.5)
                        trial_r = _score(trial)
                        if trial_r > best_r:
                            best_pts = trial
                            best_r = trial_r
                            improved = True
                            break
        else:
            # Pairwise constraint is active: push the closest pair apart
            i, j = min_pair_idx
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            if d > 1e-12:
                direction = np.array([dx/d, dy/d])
                for step in [0.001, 0.0005, 0.0001]:
                    trial = pts.copy()
                    trial[i] += step * direction
                    trial[j] -= step * direction
                    trial = np.clip(trial, -0.5, 0.5)
                    trial_r = _score(trial)
                    if trial_r > best_r:
                        best_pts = trial
                        best_r = trial_r
                        improved = True
                        break
        
        # Also try random perturbations of the active constraint points
        if not improved:
            # Perturb the active points randomly
            active_points = []
            if min_boundary_val <= min_pair_dist / 2.0:
                active_points.append(min_boundary_idx)
            else:
                active_points.extend([min_pair_idx[0], min_pair_idx[1]])
            
            for _ in range(3):
                trial = best_pts.copy()
                for idx in active_points:
                    trial[idx] += rng.normal(0, 0.002, size=2)
                trial = np.clip(trial, -0.5, 0.5)
                trial_r = _score(trial)
                if trial_r > best_r:
                    best_pts = trial
                    best_r = trial_r
                    improved = True
                    break
        
        if not improved:
            # No improvement in this iteration; add small random noise to escape local optima
            best_pts += rng.normal(0, 0.0005, size=(N, 2))
            best_pts = np.clip(best_pts, -0.5, 0.5)
            best_r = _score(best_pts)
    
    return best_pts

def _optimize(pts, time_budget):
    """
    Run L-BFGS-B optimization with multi-temperature soft-min annealing.
    """
    t0 = time.time()
    x0 = pts.flatten()
    
    # Multi-temperature schedule: 4 levels for gradual refinement
    phases = [
        {"alpha": 10.0, "max_iter": 200, "gtol": 1e-4},
        {"alpha": 30.0, "max_iter": 300, "gtol": 1e-5},
        {"alpha": 100.0, "max_iter": 400, "gtol": 1e-6},
        {"alpha": 300.0, "max_iter": 500, "gtol": 1e-7},
    ]
    
    for phase in phases:
        if time.time() - t0 > time_budget:
            break
            
        alpha = phase["alpha"]
        
        def obj(x):
            return _soft_min_neg(x, alpha)
        
        try:
            res = minimize(obj, x0, method='L-BFGS-B', 
                           bounds=[(-0.5, 0.5)] * (2 * N),
                           options={'maxiter': phase["max_iter"], 'gtol': phase["gtol"],
                                    'ftol': 1e-12})
            x_opt = res.x
            x0 = x_opt
        except Exception:
            pass
            
    # Final polish with exact objective and very tight tolerance
    remaining = time_budget - (time.time() - t0)
    if remaining > 2.0:
        try:
            res = minimize(_exact_min_neg, x0, method='L-BFGS-B', 
                           bounds=[(-0.5, 0.5)] * (2 * N),
                           options={'maxiter': 300, 'gtol': 1e-9, 'ftol': 1e-14})
            x0 = res.x
        except Exception:
            pass
            
    return x0.reshape(N, 2)

def _local_polish(pts, time_budget):
    """
    Simple local search polish: try small perturbations and keep improvements.
    """
    t0 = time.time()
    best_pts = pts.copy()
    best_r = _score(best_pts)
    
    rng = np.random.default_rng(12345)
    step_size = 0.001
    
    while time.time() - t0 < time_budget:
        perturb = rng.normal(0, step_size, size=(N, 2))
        trial = best_pts + perturb
        trial = np.clip(trial, -0.5, 0.5)
        
        trial_r = _score(trial)
        if trial_r > best_r:
            best_pts = trial
            best_r = trial_r
            step_size *= 0.95
        else:
            step_size *= 1.05
            step_size = min(step_size, 0.01)
    
    return best_pts

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
        if time.time() > DEADLINE - 20:
            break
            
        pts = init_fn()
        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
            
        remaining = DEADLINE - time.time()
        if remaining < 10:
            break
            
        # Allocate time based on initializer quality
        if name == "hex":
            time_budget = min(remaining * 0.35, 45.0)
        elif name == "grid":
            time_budget = min(remaining * 0.25, 30.0)
        else:
            time_budget = min(remaining * 0.15, 20.0)
            
        opt_pts = _optimize(pts, time_budget)
        opt_pts = np.clip(opt_pts, -0.5, 0.5)
        
        opt_r = _score(opt_pts)
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()
            
    # Targeted local search on the best solution
    remaining = DEADLINE - time.time()
    if remaining > 15 and best_pts is not None:
        targeted_budget = min(remaining * 0.3, 30.0)
        targeted_pts = _targeted_local_search(best_pts, targeted_budget)
        targeted_pts = np.clip(targeted_pts, -0.5, 0.5)
        targeted_r = _score(targeted_pts)
        if targeted_r > best_r:
            best_r = targeted_r
            best_pts = targeted_pts.copy()
    
    # Final optimization pass with remaining time
    remaining = DEADLINE - time.time()
    if remaining > 10 and best_pts is not None:
        # More L-BFGS-B optimization
        time_budget = min(remaining * 0.4, 30.0)
        opt_pts = _optimize(best_pts, time_budget)
        opt_pts = np.clip(opt_pts, -0.5, 0.5)
        opt_r = _score(opt_pts)
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()
        
        # Another targeted local search
        remaining = DEADLINE - time.time()
        if remaining > 8:
            targeted_budget = min(remaining * 0.3, 15.0)
            targeted_pts = _targeted_local_search(best_pts, targeted_budget)
            targeted_pts = np.clip(targeted_pts, -0.5, 0.5)
            targeted_r = _score(targeted_pts)
            if targeted_r > best_r:
                best_r = targeted_r
                best_pts = targeted_pts.copy()
        
        # Final random polish
        remaining = DEADLINE - time.time()
        if remaining > 5:
            time_budget = min(remaining * 0.7, 10.0)
            polished_pts = _local_polish(best_pts, time_budget)
            polished_r = _score(polished_pts)
            if polished_r > best_r:
                best_r = polished_r
                best_pts = polished_pts.copy()
    
    if best_pts is None:
        best_pts = _hex_init()
        
    return best_pts
