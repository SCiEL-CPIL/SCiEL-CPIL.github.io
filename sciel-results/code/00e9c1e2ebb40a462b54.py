"""
Robust Heilbronn Triangles Solver
Uses barycentric coordinates with strict projection to ensure points remain in the triangle.
Multi-stage optimization:
1. Repulsion phase to spread points.
2. Soft-min area maximization.
3. Hard-min area maximization (polish).
"""

import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2.0
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])
V = np.array([V0, V1, V2])

# Precompute triangle indices for vectorized area computation
_TRI_IDX = np.array(list(combinations(range(N), 3)))
_N_TRI = _TRI_IDX.shape[0]

# Precompute pair indices for distance penalty
_PAIR_IDX = np.array(list(combinations(range(N), 2)))
_N_PAIR = _PAIR_IDX.shape[0]

def _project_to_triangle(bary_flat):
    """
    Project barycentric coordinates (u, v) to the valid triangle domain.
    Valid domain: u >= 0, v >= 0, u + v <= 1.
    Input: flat array of length 2*N (u0, v0, u1, v1, ...)
    Output: projected flat array.
    """
    u = bary_flat[0::2].copy()
    v = bary_flat[1::2].copy()
    
    # Clip to non-negative
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    
    # Enforce u + v <= 1
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        # Scale down proportionally to hit the boundary u+v=1
        u[mask] /= s[mask]
        v[mask] /= s[mask]
        
    return np.concatenate([u, v])

def _bary_to_cart(bary_flat):
    """
    Convert flat barycentric array (2*N,) to Cartesian points (N, 2).
    Input bary_flat[i] = u_i, bary_flat[i+1] = v_i.
    w_i = 1 - u_i - v_i.
    """
    # Ensure validity just in case
    bary_flat = _project_to_triangle(bary_flat)
    u = bary_flat[0::2]
    v = bary_flat[1::2]
    w = 1.0 - u - v
    
    pts = np.zeros((N, 2))
    pts += u[:, None] * V0
    pts += v[:, None] * V1
    pts += w[:, None] * V2
    return pts

def _all_areas_cart(pts):
    """Vectorized area computation for all triangles."""
    pi = pts[_TRI_IDX[:, 0]]
    pj = pts[_TRI_IDX[:, 1]]
    pk = pts[_TRI_IDX[:, 2]]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)
    return areas

def _all_dists_cart(pts):
    """Vectorized squared distance computation for all pairs."""
    pi = pts[_PAIR_IDX[:, 0]]
    pj = pts[_PAIR_IDX[:, 1]]
    dx = pj[:, 0] - pi[:, 0]
    dy = pj[:, 1] - pi[:, 1]
    dists_sq = dx**2 + dy**2
    return dists_sq

def _objective(bary_flat, beta, penalty_weight, time_start, deadline):
    """
    Objective function for L-BFGS-B.
    Minimizes: -soft_min_area + penalty_weight * sum(1/d^2)
    """
    if time.time() > deadline:
        return 1e10

    # Project to valid domain to ensure smoothness and validity
    bary_flat = _project_to_triangle(bary_flat)
    pts = _bary_to_cart(bary_flat)
    
    # Calculate areas
    areas = _all_areas_cart(pts)
    
    # Soft min of areas
    max_a = np.max(areas)
    if max_a < 1e-12:
        return 1e10
        
    exp_terms = np.exp(-beta * (areas - max_a))
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = -(max_a + log_sum / beta)
    
    obj_area = -soft_min_val
    
    # Calculate distance penalty
    dists_sq = _all_dists_cart(pts)
    dists_sq = np.maximum(dists_sq, 1e-12)
    penalty = np.sum(1.0 / dists_sq)
    
    return obj_area + penalty_weight * penalty

def _hard_objective(bary_flat, time_start, deadline):
    """
    Objective for final polish: maximize min area.
    Since we minimize, return -min_area.
    """
    if time.time() > deadline:
        return 1e10

    bary_flat = _project_to_triangle(bary_flat)
    pts = _bary_to_cart(bary_flat)
    areas = _all_areas_cart(pts)
    return -np.min(areas)

def _generate_initial_bary(seed):
    """Generate a random valid barycentric configuration."""
    rng = np.random.default_rng(seed)
    # Generate random points in triangle using sqrt method for uniform distribution
    r1 = rng.random(N)
    r2 = rng.random(N)
    sqrt_r1 = np.sqrt(r1)
    u = sqrt_r1 * (1 - r2)
    v = sqrt_r1 * r2
    return np.concatenate([u, v])

def _generate_grid_bary():
    """Generate a structured grid-like initial configuration."""
    pts_bary = []
    # Center
    pts_bary.append([1/3, 1/3])
    # Ring 1
    for i in range(6):
        angle = i * np.pi / 3
        r = 0.2
        # Barycentric coordinates for a point in triangle are tricky to map directly from polar.
        # Instead, let's just use a few fixed good points.
        pass
    
    # Let's use a simpler deterministic spread
    # Vertices (inset)
    eps = 0.1
    pts_bary.append([1-eps, eps/2]) # Near V0? No, V0 is (1,0) in bary (u,v) if w=0? 
    # Recall: P = u*V0 + v*V1 + w*V2.
    # V0=(0,0) -> u=1, v=0.
    # V1=(1,0) -> u=0, v=1.
    # V2=(0.5,H) -> u=0, v=0.
    
    # Inset vertices
    pts_bary.append([0.8, 0.1]) # Near V0
    pts_bary.append([0.1, 0.8]) # Near V1
    pts_bary.append([0.1, 0.1]) # Near V2
    
    # Midpoints
    pts_bary.append([0.4, 0.4]) # Mid V0-V1
    pts_bary.append([0.4, 0.0]) # Mid V0-V2
    pts_bary.append([0.0, 0.4]) # Mid V1-V2
    
    # Interior
    pts_bary.append([1/3, 1/3])
    pts_bary.append([0.3, 0.3])
    pts_bary.append([0.2, 0.2])
    pts_bary.append([0.5, 0.2])
    
    arr = np.array(pts_bary).flatten()
    return _project_to_triangle(arr)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_bary = None
    best_score = -1.0
    
    # Initializations
    inits = [
        _generate_grid_bary(),
        _generate_initial_bary(42),
        _generate_initial_bary(123),
        _generate_initial_bary(789),
        _generate_initial_bary(101),
        _generate_initial_bary(555),
        _generate_initial_bary(2024),
    ]
    
    # Optimization Stages
    # Stage 1: High repulsion, low beta (smooth landscape, spread out)
    # Stage 2: Medium repulsion, medium beta
    # Stage 3: Low repulsion, high beta (precise max-min)
    # Stage 4: No repulsion, very high beta (final polish)
    
    stages = [
        (5.0, 50.0, 100),   # beta, penalty, maxiter
        (20.0, 10.0, 150),
        (100.0, 1.0, 200),
        (500.0, 0.1, 300),
        (2000.0, 0.0, 500), # Final high precision
    ]
    
    bounds = [(0.0, 1.0)] * (2 * N)
    
    for init_idx, init_bary in enumerate(inits):
        if time.time() > deadline - 15:
            break
            
        x_current = init_bary.copy()
        
        for stage_idx, (beta, pw, maxiter) in enumerate(stages):
            if time.time() > deadline - 5:
                break
                
            try:
                # Use L-BFGS-B. The objective projects the input, so it handles constraints.
                res = minimize(
                    _objective,
                    x_current,
                    args=(beta, pw, start_time, deadline),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                # Check if result is better
                current_obj = _objective(x_current, beta, pw, start_time, deadline)
                res_obj = _objective(res.x, beta, pw, start_time, deadline)
                
                if res_obj < current_obj:
                    x_current = res.x
                elif res.success:
                    x_current = res.x
            except Exception:
                pass
        
        # Evaluate final min area for this run
        pts = _bary_to_cart(x_current)
        areas = _all_areas_cart(pts)
        min_a = np.min(areas)
        
        if min_a > best_score:
            best_score = min_a
            best_bary = x_current.copy()
            
    # Final Polish on the best found so far using Hard Min Objective
    if best_bary is not None and time.time() < deadline - 2.0:
        try:
            res = minimize(
                _hard_objective,
                best_bary,
                args=(start_time, deadline),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.success:
                best_bary = res.x
        except Exception:
            pass

    # Convert best barycentric to Cartesian
    if best_bary is None:
        # Fallback
        best_bary = _generate_grid_bary()
        
    # Final strict projection and conversion
    best_bary = _project_to_triangle(best_bary)
    final_pts = _bary_to_cart(best_bary)
    
    # Double check validity
    # Barycentric coords u, v, w=1-u-v must be >= 0
    # Our projection ensures this.
    
    return final_pts

# EVOLVE-BLOCK-END
