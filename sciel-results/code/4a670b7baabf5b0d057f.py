"""
Heilbronn triangles: Maximize minimum triangle area for 11 points.
Hybrid approach: Simulated Annealing (Global) + L-BFGS-B with Log-Sum-Exp (Local).
"""
N = 11

import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

_SQRT3 = np.sqrt(3.0)
_H = _SQRT3 / 2.0
_EPS = 1e-9

# Precompute all C(11,3) = 165 index triples
_IDX = np.array(list(combinations(range(N), 3)))
_I = _IDX[:, 0]
_J = _IDX[:, 1]
_K = _IDX[:, 2]

def _cart_to_bary(pts):
    """Convert Cartesian points (x,y) to Barycentric (u,v) where w=1-u-v."""
    x = pts[:, 0]
    y = pts[:, 1]
    u = 1.0 - x - y / _SQRT3
    v = x - y / _SQRT3
    return np.column_stack((u, v))

def _bary_to_cart(bary):
    """Convert Barycentric (u,v) to Cartesian (x,y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    x = 0.5 - 0.5 * u + 0.5 * v
    y = _H * (1.0 - u - v)
    return np.column_stack((x, y))

def _clamp_cart(pts):
    """Clamp Cartesian points to the triangle."""
    pts = np.array(pts, dtype=float).copy()
    x = pts[:, 0]
    y = pts[:, 1]
    y = np.clip(y, _EPS, _H - _EPS)
    x_min = y / _SQRT3 + _EPS
    x_max = 1.0 - y / _SQRT3 - _EPS
    x = np.clip(x, x_min, x_max)
    pts[:, 0] = x
    pts[:, 1] = y
    return pts

def _areas_vec(pts):
    """Calculate all 165 triangle areas."""
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _soft_min_area(pts, beta):
    """Calculate soft-min of areas using Log-Sum-Exp."""
    areas = _areas_vec(pts)
    # Avoid log(0)
    areas = np.maximum(areas, 1e-12)
    x_vals = -beta * areas
    m = np.max(x_vals)
    if m < -700: # Underflow protection
        return -1.0 / beta
    log_sum = m + np.log(np.sum(np.exp(x_vals - m)))
    return -log_sum / beta

def _min_area(pts):
    """Exact minimum area."""
    return float(np.min(_areas_vec(pts)))

def _sa_objective(pts, beta):
    """Objective for SA: negative soft-min area."""
    return -_soft_min_area(pts, beta)

def _lbfgsb_objective(bary_flat, beta, penalty_weight):
    """
    Objective for L-BFGS-B.
    Input: flattened barycentric coordinates (2N).
    Output: negative soft-min area + penalty.
    """
    bary = bary_flat.reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    
    # Penalty for u+v > 1 (points outside triangle)
    # u, v are bounded to [0,1] by L-BFGS-B, so u>=0, v>=0 is handled.
    viol = np.maximum(0.0, u + v - 1.0)
    penalty = penalty_weight * np.sum(viol**2)
    
    # Convert to Cartesian to calculate areas
    pts = _bary_to_cart(bary)
    
    # Calculate soft-min area
    sm = _soft_min_area(pts, beta)
    
    return -sm + penalty

def _run_sa(pts_init, time_limit, beta=50.0, step_size=0.05, cooling_rate=0.99):
    """
    Run Simulated Annealing.
    """
    start_time = time.time()
    pts = _clamp_cart(np.array(pts_init, dtype=float))
    current_score = _sa_objective(pts, beta)
    best_pts = pts.copy()
    best_score = current_score
    
    temperature = 1.0
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < time_limit:
        # Perturb
        new_pts = pts + rng.normal(0, step_size, (N, 2))
        new_pts = _clamp_cart(new_pts)
        
        new_score = _sa_objective(new_pts, beta)
        
        # Metropolis criterion
        delta = new_score - current_score
        if delta > 0 or rng.random() < np.exp(delta / temperature):
            pts = new_pts
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
        
        # Cool down
        temperature *= cooling_rate
        
        # Reset step size if stuck? 
        # Simple adaptive: if no improvement for a while, reduce step size
        # For now, fixed cooling is okay for short runs.
        
    return best_pts, best_score

def _run_lbfgsb(bary_init, time_limit, beta=500.0, penalty_weight=1000.0):
    """
    Run L-BFGS-B optimization on barycentric coordinates.
    """
    start_time = time.time()
    x0 = np.array(bary_init, dtype=float).flatten()
    
    # Bounds for u, v in [0, 1]
    bounds = [(0.0, 1.0)] * (2 * N)
    
    best_x = x0.copy()
    best_obj = _lbfgsb_objective(best_x, beta, penalty_weight)
    
    try:
        res = minimize(
            _lbfgsb_objective,
            x0,
            args=(beta, penalty_weight),
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 500,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxfun': 5000,
            }
        )
        
        cand_x = res.x
        cand_obj = _lbfgsb_objective(cand_x, beta, penalty_weight)
        
        if cand_obj < best_obj:
            best_obj = cand_obj
            best_x = cand_x
            
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time < time_limit * 0.8:
        # Try with higher beta for sharper min
        try:
            res2 = minimize(
                _lbfgsb_objective,
                best_x,
                args=(2000.0, penalty_weight * 10),
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 300,
                    'ftol': 1e-14,
                    'gtol': 1e-10,
                }
            )
            cand_x2 = res2.x
            cand_obj2 = _lbfgsb_objective(cand_x2, 2000.0, penalty_weight * 10)
            
            # Compare using the same beta scale? 
            # Actually, just check the resulting min area
            pts2 = _bary_to_cart(cand_x2.reshape(N, 2))
            pts2 = _clamp_cart(pts2)
            area2 = _min_area(pts2)
            
            pts_best = _bary_to_cart(best_x.reshape(N, 2))
            pts_best = _clamp_cart(pts_best)
            area_best = _min_area(pts_best)
            
            if area2 > area_best:
                best_x = cand_x2
                
        except Exception:
            pass

    return _bary_to_cart(best_x.reshape(N, 2))

def _generate_seeds():
    """Generate diverse initial configurations in Cartesian."""
    h = _H
    seeds = []
    
    # Seed 1: Vertices + centroid + hexagonal
    s1 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.5, h/3],
        [0.25, h/6],
        [0.75, h/6],
        [0.5, h/2],
        [0.167, h/12],
        [0.833, h/12],
        [0.333, h/4],
        [0.667, h/4],
    ])
    seeds.append(_clamp_cart(s1))
    
    # Seed 2: Triangular grid
    s2 = np.array([
        [0.0, 0.0],
        [0.5, 0.0],
        [1.0, 0.0],
        [0.25, h/4],
        [0.5, h/4],
        [0.75, h/4],
        [0.5, h/2],
        [0.25, h/2],
        [0.75, h/2],
        [0.5, 2*h/3],
        [0.25, 2*h/3],
    ])
    seeds.append(_clamp_cart(s2))
    
    # Seed 3: Random
    rng = np.random.default_rng(123)
    for _ in range(5):
        pts = rng.random((N, 2))
        pts[:, 1] *= h
        seeds.append(_clamp_cart(pts))
        
    return seeds

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_pts = None
    best_score = -1.0
    
    def _update(pts, score):
        nonlocal best_pts, best_score
        if score > best_score:
            best_score = score
            best_pts = np.array(pts, dtype=float).copy()
    
    seeds = _generate_seeds()
    
    # Phase 1: SA + L-BFGS-B for each seed
    # We have ~140s. 
    # Let's allocate time per seed.
    num_seeds = len(seeds)
    time_per_seed = 12.0 # 5 seeds * 12s = 60s
    
    for i, seed in enumerate(seeds):
        if time.time() > deadline - 10:
            break
            
        # 1. SA Phase (6s)
        sa_time = 6.0
        sa_pts, _ = _run_sa(seed, time_limit=sa_time, beta=50.0, step_size=0.05, cooling_rate=0.995)
        
        # 2. L-BFGS-B Phase (5s)
        # Convert to barycentric
        bary_init = _cart_to_bary(sa_pts)
        # Ensure barycentric coords are valid for L-BFGS-B bounds [0,1]
        # u = 1 - x - y/sqrt(3), v = x - y/sqrt(3)
        # If points are clamped to triangle, u,v >= 0 and u+v <= 1.
        # However, due to floating point, u or v might be slightly negative or u+v > 1.
        # L-BFGS-B will handle bounds [0,1]. The penalty handles u+v>1.
        
        lbfgsb_time = 5.0
        lbfgsb_pts = _run_lbfgsb(bary_init, time_limit=lbfgsb_time, beta=500.0, penalty_weight=1000.0)
        
        # Evaluate
        score = _min_area(lbfgsb_pts)
        _update(lbfgsb_pts, score)
        
        # 3. Refine SA result directly?
        # The L-BFGS-B result is usually better.
        
    # Phase 2: Polish the best result found
    if best_pts is not None and time.time() < deadline - 5:
        remaining = deadline - 5 - time.time()
        if remaining > 2:
            # Run L-BFGS-B again on the best points with higher beta
            bary_best = _cart_to_bary(best_pts)
            lbfgsb_pts = _run_lbfgsb(bary_best, time_limit=min(10.0, remaining), beta=2000.0, penalty_weight=10000.0)
            score = _min_area(lbfgsb_pts)
            _update(lbfgsb_pts, score)

    if best_pts is None:
        best_pts = seeds[0]
        
    return _clamp_cart(best_pts)
