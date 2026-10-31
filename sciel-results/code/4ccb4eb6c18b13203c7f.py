"""
Heilbronn triangles: Maximize minimum triangle area for 11 points.
Hybrid approach: Direct Barycentric Nelder-Mead + Multi-start.
Uses barycentric coordinates throughout to avoid conversion overhead.
"""
N = 11

import numpy as np
import time
from itertools import combinations

_SQRT3 = np.sqrt(3.0)
_H = _SQRT3 / 2.0
_EPS = 1e-9

# Precompute all C(11,3) = 165 index triples
_IDX = np.array(list(combinations(range(N), 3)))
_I = _IDX[:, 0]
_J = _IDX[:, 1]
_K = _IDX[:, 2]

def _bary_to_cart(bary):
    """Convert Barycentric (u,v) to Cartesian (x,y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    x = 0.5 - 0.5 * u + 0.5 * v
    y = _H * (1.0 - u - v)
    return np.column_stack((x, y))

def _cart_to_bary(pts):
    """Convert Cartesian points (x,y) to Barycentric (u,v) where w=1-u-v."""
    x = pts[:, 0]
    y = pts[:, 1]
    u = 1.0 - x - y / _SQRT3
    v = x - y / _SQRT3
    return np.column_stack((u, v))

def _clamp_bary(bary, eps=1e-8):
    """
    Clamp barycentric coordinates to the strictly valid triangle region.
    u >= eps, v >= eps, u+v <= 1 - eps
    """
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    # Clip to strict interior
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    
    # Handle u + v > 1 constraint
    excess = u + v - 1.0
    mask = excess > 0
    if np.any(mask):
        # Reduce both u and v proportionally
        scale = 1.0 / (1.0 + excess[mask])
        u[mask] = u[mask] * scale
        v[mask] = v[mask] * scale
    
    return np.column_stack((u, v))

def _calc_areas_bary(bary):
    """
    Calculate all 165 triangle areas directly from barycentric coordinates.
    For points with barycentric coords (u,v) where w=1-u-v, the area of triangle
    formed by points i, j, k can be computed using the determinant formula.
    """
    # Convert to Cartesian for area calculation (most straightforward)
    pts = _bary_to_cart(bary)
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _soft_min_obj(bary, beta=100.0):
    """
    Objective for optimization: negative soft-min of triangle areas.
    Lower is better (we minimize this).
    """
    areas = _calc_areas_bary(bary)
    areas = np.maximum(areas, 1e-12)
    x_vals = -beta * areas
    m = np.max(x_vals)
    if m < -700:
        return 1.0 / beta
    log_sum = m + np.log(np.sum(np.exp(x_vals - m)))
    return log_sum / beta  # Note: sign is positive since we're minimizing

def _hard_min_obj(bary):
    """
    Objective for evaluation: negative hard-min of triangle areas.
    """
    areas = _calc_areas_bary(bary)
    return -np.min(areas)

def _nelder_mead_polish(bary_init, max_iter=2000, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5, time_limit=10.0):
    """
    Custom Nelder-Mead implementation for barycentric coordinates.
    Works directly in barycentric space with constraint clamping.
    """
    start_time = time.time()
    
    bary = _clamp_bary(bary_init.copy())
    n = bary.size  # 2N = 22
    
    # Initialize simplex
    simplex = [bary.flatten().copy()]
    f_vals = [_soft_min_obj(bary)]
    
    # Create initial simplex by perturbing each coordinate
    for i in range(n):
        point = bary.flatten().copy()
        if point[i] == 0:
            point[i] = 0.01
        else:
            point[i] *= 1.05
        simplex.append(point)
        f_vals.append(_soft_min_obj(np.array(simplex[-1]).reshape(N, 2)))
    
    # Sort simplex by objective value
    order = np.argsort(f_vals)
    simplex = [simplex[i] for i in order]
    f_vals = [f_vals[i] for i in order]
    
    best_bary = np.array(simplex[0]).reshape(N, 2)
    best_f = f_vals[0]
    
    it = 0
    while it < max_iter and time.time() - start_time < time_limit:
        it += 1
        
        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        xr_bary = _clamp_bary(xr.reshape(N, 2))
        fr = _soft_min_obj(xr_bary)
        
        if fr < f_vals[0]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
            xe_bary = _clamp_bary(xe.reshape(N, 2))
            fe = _soft_min_obj(xe_bary)
            
            if fe < fr:
                simplex[-1] = xe_bary.flatten()
                f_vals[-1] = fe
            else:
                simplex[-1] = xr_bary.flatten()
                f_vals[-1] = fr
        elif fr < f_vals[-2]:
            simplex[-1] = xr_bary.flatten()
            f_vals[-1] = fr
        else:
            # Contraction
            if fr < f_vals[-1]:
                # Outside contraction
                xc = centroid + rho * (xr - centroid)
            else:
                # Inside contraction
                xc = centroid + rho * (simplex[-1] - centroid)
            
            xc_bary = _clamp_bary(xc.reshape(N, 2))
            fc = _soft_min_obj(xc_bary)
            
            if fc < f_vals[-1]:
                simplex[-1] = xc_bary.flatten()
                f_vals[-1] = fc
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = _clamp_bary(simplex[i].reshape(N, 2)).flatten()
                    f_vals[i] = _soft_min_obj(simplex[i].reshape(N, 2))
        
        # Sort simplex
        order = np.argsort(f_vals)
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]
        
        # Update best
        if f_vals[0] < best_f:
            best_f = f_vals[0]
            best_bary = np.array(simplex[0]).reshape(N, 2)
    
    return best_bary

def _generate_hexagonal_init():
    """
    Generate a structured hexagonal-like initial configuration.
    """
    h = _H
    # Start with vertices and add interior points in a pattern
    bary = np.array([
        [0.0, 0.0],      # Vertex 1
        [1.0, 0.0],      # Vertex 2
        [0.0, 1.0],      # Vertex 3
        [1/3, 1/3],      # Centroid
        [1/2, 1/6],      # Edge midpoint 1
        [1/6, 1/2],      # Edge midpoint 2
        [1/6, 1/6],      # Corner interior 1
        [5/6, 1/6],      # Corner interior 2
        [1/6, 5/6],      # Corner interior 3
        [1/2, 1/3],      # Interior 1
        [1/3, 1/2],      # Interior 2
    ])
    return _clamp_bary(bary)

def _generate_random_init(seed=42):
    """
    Generate a random initial configuration in barycentric coordinates.
    """
    rng = np.random.default_rng(seed)
    # Sample from uniform distribution in triangle
    u = rng.random(N)
    v = rng.random(N)
    mask = u + v > 1.0
    u[mask], v[mask] = v[mask], u[mask]
    # This gives uniform distribution in the triangle
    bary = np.column_stack((u, v))
    return _clamp_bary(bary)

def _generate_perturbed_init(base, seed=42, scale=0.1):
    """
    Generate a perturbed version of a base configuration.
    """
    rng = np.random.default_rng(seed)
    bary = base + rng.normal(0, scale, base.shape)
    return _clamp_bary(bary)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    best_bary = None
    best_score = -np.inf
    
    def _update(bary, score):
        nonlocal best_bary, best_score
        if score > best_score:
            best_score = score
            best_bary = bary.copy()
    
    # Generate diverse initial configurations
    seeds = [
        _generate_hexagonal_init(),
        _generate_random_init(123),
        _generate_random_init(456),
        _generate_random_init(789),
        _generate_perturbed_init(_generate_hexagonal_init(), 111, 0.05),
        _generate_perturbed_init(_generate_hexagonal_init(), 222, 0.1),
    ]
    
    # Phase 1: Nelder-Mead on each seed
    time_per_seed = 15.0  # 6 seeds * 15s = 90s
    
    for i, seed in enumerate(seeds):
        if time.time() > deadline - 10:
            break
        
        # Run Nelder-Mead with time limit
        time_limit = min(time_per_seed, deadline - 10 - time.time())
        if time_limit < 1:
            break
        
        polished = _nelder_mead_polish(seed, max_iter=1000, time_limit=time_limit)
        
        # Evaluate with hard min
        score = -_hard_min_obj(polished)
        _update(polished, score)
        
        # If time allows, do a second polish with higher beta
        if time.time() < deadline - 5:
            remaining = deadline - 5 - time.time()
            if remaining > 3:
                polished2 = _nelder_mead_polish(polished, max_iter=500, time_limit=min(5.0, remaining))
                score2 = -_hard_min_obj(polished2)
                _update(polished2, score2)
    
    # Phase 2: Final polish on best result
    if best_bary is not None and time.time() < deadline - 3:
        remaining = deadline - 3 - time.time()
        if remaining > 2:
            final_polish = _nelder_mead_polish(best_bary, max_iter=500, time_limit=min(5.0, remaining))
            final_score = -_hard_min_obj(final_polish)
            _update(final_polish, final_score)
    
    if best_bary is None:
        best_bary = _generate_hexagonal_init()
    
    # Convert to Cartesian for output
    return _bary_to_cart(_clamp_bary(best_bary))
