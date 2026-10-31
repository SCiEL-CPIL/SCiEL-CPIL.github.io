"""
Heilbronn triangles: Maximize minimum triangle area for 11 points.
Hybrid approach: Barycentric L-BFGS-B with Log-Sum-Exp (Local) + Nelder-Mead Polish.
Incorporates direct barycentric area calculation for speed and stability.
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

def _clamp_bary(bary, eps=1e-7):
    """
    Clamp barycentric coordinates to the strictly valid triangle region.
    u >= eps, v >= eps, u+v <= 1 - eps
    """
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    # Clip to strict interior
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    # Ensure u + v <= 1 - eps
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    # Re-check bounds after scaling
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def _areas_bary(bary):
    """
    Calculate areas of all triangles formed by N points given in barycentric coords (u, v).
    Returns array of shape (165,) containing the area ratios (relative to total triangle area).
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    ui = u[_I]
    vi = v[_I]
    wi = w[_I]
    
    uj = u[_J]
    vj = v[_J]
    wj = w[_J]
    
    uk = u[_K]
    vk = v[_K]
    wk = w[_K]
    
    # Determinant calculation vectorized
    # det = ui*(vj*wk - vk*wj) - vi*(uj*wk - uk*wj) + wi*(uj*vk - uk*vj)
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def _soft_min_area_bary(bary, beta):
    """Calculate soft-min of areas using Log-Sum-Exp."""
    areas = _areas_bary(bary)
    # Avoid log(0)
    areas = np.maximum(areas, 1e-12)
    x_vals = -beta * areas
    m = np.max(x_vals)
    if m < -700: # Underflow protection
        return -1.0 / beta
    log_sum = m + np.log(np.sum(np.exp(x_vals - m)))
    return -log_sum / beta

def _min_area_bary(bary):
    """Exact minimum area."""
    return float(np.min(_areas_bary(bary)))

def _bary_to_cart(bary):
    """
    Convert Barycentric coords (u, v) to Cartesian (x, y).
    bary: (N, 2) array of (u, v). w = 1 - u - v.
    Point = u*A + v*B + w*C
    A=(0,0), B=(1,0), C=(0.5, H)
    x = v + 0.5*w
    y = w*H
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * _H
    return np.column_stack([x, y])

def _generate_seeds():
    """Generate diverse initial configurations in Barycentric coordinates."""
    seeds = []
    
    # Seed 1: Hexagonal/Lattice-like in Barycentric
    # Try a few scales
    for s in [0.2, 0.25, 0.3, 0.4]:
        pts = []
        # Simple grid in u-v space
        for i in range(5):
            for j in range(5):
                u = i * s
                v = j * s * _SQRT3 / 2
                if u + v <= 1.0 and len(pts) < N:
                    pts.append([u, v])
        if len(pts) == N:
            seeds.append(np.array(pts))
            
    # Seed 2: Random
    rng = np.random.default_rng(123)
    for _ in range(5):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        seeds.append(np.column_stack([u, v]))
        
    # Seed 3: Centroid + Vertices + Midpoints (approx)
    # Vertices: (1,0), (0,1), (0,0) in bary (u,v) where w=1-u-v
    # Actually vertices are (1,0) [A], (0,1) [B], (0,0) [C]
    s3 = np.array([
        [1.0, 0.0], # A
        [0.0, 1.0], # B
        [0.0, 0.0], # C
        [1/3, 1/3], # Centroid
        [0.5, 0.0], # Mid AB
        [0.0, 0.5], # Mid BC
        [0.5, 0.5], # Mid CA? No, u=0.5, v=0.5 -> w=0. Mid of C and... wait.
                    # C is (0,0). Mid AC is (0.5, 0). Mid BC is (0, 0.5).
                    # Mid AB is (0.5, 0.5)? No.
                    # A=(1,0), B=(0,1). Mid = (0.5, 0.5). w=0.
                    # So [0.5, 0.5] is Mid AB.
                    # Let's use [0.5, 0.0] (Mid AC), [0.0, 0.5] (Mid BC), [0.5, 0.5] (Mid AB)
        [0.25, 0.25], # Quarter
        [0.75, 0.25],
        [0.25, 0.75],
        [0.5, 0.25],
    ])
    # Clamp to ensure validity
    seeds.append(_clamp_bary(s3))
    
    return seeds

def _run_lbfgsb(bary_init, time_limit, beta=500.0):
    """
    Run L-BFGS-B optimization on barycentric coordinates.
    """
    start_time = time.time()
    x0 = np.array(bary_init, dtype=float).flatten()
    
    # Bounds for u, v in [0, 1]
    bounds = [(0.0, 1.0)] * (2 * N)
    
    best_x = x0.copy()
    best_obj = -_soft_min_area_bary(x0.reshape(N,2), beta)
    
    def obj(x):
        b = _clamp_bary(x.reshape(N, 2))
        return -_soft_min_area_bary(b, beta)
        
    try:
        res = minimize(
            obj,
            x0,
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
        cand_obj = obj(cand_x)
        
        if cand_obj < best_obj:
            best_obj = cand_obj
            best_x = cand_x.copy()
            
    except Exception:
        pass
    
    # Check time for second stage
    if time.time() - start_time < time_limit * 0.8:
        # Try with higher beta for sharper min
        beta2 = 2000.0
        def obj2(x):
            b = _clamp_bary(x.reshape(N, 2))
            return -_soft_min_area_bary(b, beta2)
            
        try:
            res2 = minimize(
                obj2,
                best_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 300,
                    'ftol': 1e-14,
                    'gtol': 1e-10,
                }
            )
            cand_x2 = res2.x
            # Compare using actual min area
            pts2 = _clamp_bary(cand_x2.reshape(N, 2))
            area2 = _min_area_bary(pts2)
            
            pts_best = _clamp_bary(best_x.reshape(N, 2))
            area_best = _min_area_bary(pts_best)
            
            if area2 > area_best:
                best_x = cand_x2.copy()
                
        except Exception:
            pass

    return _clamp_bary(best_x.reshape(N, 2))

def _run_nelder_mead(bary_init, time_limit):
    """
    Run Nelder-Mead on the hard min objective.
    """
    start_time = time.time()
    x0 = np.array(bary_init, dtype=float).flatten()
    
    def obj(x):
        b = _clamp_bary(x.reshape(N, 2))
        return -_min_area_bary(b)
        
    best_x = x0.copy()
    best_obj = obj(x0)
    
    try:
        res = minimize(
            obj,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 2000,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True,
            }
        )
        
        cand_x = res.x
        cand_obj = obj(cand_x)
        
        if cand_obj < best_obj:
            best_obj = cand_obj
            best_x = cand_x.copy()
            
    except Exception:
        pass
        
    return _clamp_bary(best_x.reshape(N, 2))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_bary = None
    best_score = -1.0
    
    def _update(bary, score):
        nonlocal best_bary, best_score
        if score > best_score:
            best_score = score
            best_bary = np.array(bary, dtype=float).copy()
    
    seeds = _generate_seeds()
    
    # Phase 1: L-BFGS-B for each seed
    num_seeds = len(seeds)
    # Allocate time per seed
    # We have ~140s. Let's say 10s per seed for ~14 seeds.
    time_per_seed = 10.0
    
    for i, seed in enumerate(seeds):
        if time.time() > deadline - 20:
            break
            
        lbfgsb_pts = _run_lbfgsb(seed, time_limit=time_per_seed, beta=500.0)
        score = _min_area_bary(lbfgsb_pts)
        _update(lbfgsb_pts, score)
        
    # Phase 2: Polish the best result with Nelder-Mead
    if best_bary is not None and time.time() < deadline - 10:
        remaining = deadline - 10 - time.time()
        if remaining > 5:
            nm_pts = _run_nelder_mead(best_bary, time_limit=min(30.0, remaining))
            score = _min_area_bary(nm_pts)
            _update(nm_pts, score)
            
            # Maybe one more L-BFGS-B pass on the NM result?
            if time.time() < deadline - 5:
                lbfgsb_pts2 = _run_lbfgsb(nm_pts, time_limit=min(10.0, deadline - 5 - time.time()), beta=2000.0)
                score2 = _min_area_bary(lbfgsb_pts2)
                _update(lbfgsb_pts2, score2)

    if best_bary is None:
        best_bary = seeds[0]
        
    return _bary_to_cart(best_bary)
