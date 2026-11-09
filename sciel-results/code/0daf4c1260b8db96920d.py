"""Improved hexagon packing: minimize outer hexagon side length for 11 unit hexagons."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11
_START_TIME = time.time()
TIME_LIMIT = 135.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return vertices of a regular hexagon centered at (cx,cy)."""
    angle_rad = np.deg2rad(angle_deg)
    base_angles = np.arange(6) * np.pi / 3.0
    angles = base_angles + angle_rad
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _point_in_hexagon(px, py, hex_verts):
    """Check if point is inside convex polygon."""
    n = len(hex_verts)
    for i in range(n):
        x1, y1 = hex_verts[i]
        x2, y2 = hex_verts[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if cross < -1e-9:
            return False
    return True

def _polygons_overlap_fast(v1, v2, tol=1e-6):
    """Fast SAT check for two convex hexagons. Returns True if overlapping."""
    # SAT: check all edge normals of both polygons
    for poly in (v1, v2):
        n = len(poly)
        for j in range(n):
            x1, y1 = poly[j]
            x2, y2 = poly[(j + 1) % n]
            # Edge normal
            nx = -(y2 - y1)
            ny = x2 - x1
            # Project
            p1 = v1[:, 0] * nx + v1[:, 1] * ny
            p2 = v2[:, 0] * nx + v2[:, 1] * ny
            if p1.max() < p2.min() - tol or p2.max() < p1.min() - tol:
                return False
    return True

def _min_separation(v1, v2):
    """Minimum separation distance between two convex polygons (0 if overlapping)."""
    min_sep = np.inf
    for poly in (v1, v2):
        n = len(poly)
        for j in range(n):
            x1, y1 = poly[j]
            x2, y2 = poly[(j + 1) % n]
            ex = x2 - x1
            ey = y2 - y1
            norm = np.sqrt(ex*ex + ey*ey)
            if norm < 1e-12:
                continue
            nx = -ey / norm
            ny = ex / norm
            p1 = v1[:, 0] * nx + v1[:, 1] * ny
            p2 = v2[:, 0] * nx + v2[:, 1] * ny
            sep = max(p2.min() - p1.max(), p1.min() - p2.max())
            if sep > 0:
                min_sep = min(min_sep, sep)
    return 0.0 if min_sep == np.inf else min_sep

def _get_inner_verts(params):
    """Get all inner hexagon vertices from params array."""
    verts_list = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        ang = params[3*i+2]
        verts_list.append(_hex_vertices(x, y, ang, 1.0))
    return verts_list

def _compute_penalty(params, outer_side, outer_center, outer_angle):
    """Compute penalty for invalid packing."""
    cx, cy = outer_center
    outer_verts = _hex_vertices(cx, cy, outer_angle, outer_side)
    
    penalty = 0.0
    inner_verts = _get_inner_verts(params)
    
    # Containment check
    for i in range(N):
        for px, py in inner_verts[i]:
            if not _point_in_hexagon(px, py, outer_verts):
                penalty += 5.0
    
    # Pairwise overlap check
    for i in range(N):
        for j in range(i+1, N):
            if _polygons_overlap_fast(inner_verts[i], inner_verts[j], tol=-1e-4):
                # Overlapping
                penalty += 10.0
    
    return penalty

def _objective(params, outer_side, outer_center, outer_angle):
    """Objective function: outer_side + penalty."""
    penalty = _compute_penalty(params, outer_side, outer_center, outer_angle)
    return outer_side + penalty

def _random_init(scale=3.0, seed=None):
    """Random initial configuration."""
    if seed is not None:
        np.random.seed(seed)
    params = np.zeros(3 * N)
    for i in range(N):
        params[3*i] = np.random.uniform(-scale, scale)
        params[3*i+1] = np.random.uniform(-scale, scale)
        params[3*i+2] = np.random.uniform(0, 60)
    return params

def _nm_optimize(x0, outer_side, outer_center, outer_angle, maxiter=100, timeout_fn=None):
    """Nelder-Mead optimization with timeout check."""
    def obj(p):
        if timeout_fn and timeout_fn():
            return 1e10
        return _objective(p, outer_side, outer_center, outer_angle)
    
    try:
        res = minimize(obj, x0, method='Nelder-Mead',
                      options={'maxiter': maxiter, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
        return res.x, res.fun
    except Exception:
        return x0, _objective(x0, outer_side, outer_center, outer_angle)

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    best_score = 40.0
    best_params = _random_init(2.0, seed=42)
    best_valid_outer = None
    best_valid_params = None
    
    # Multi-start: try various outer sizes and random inits
    # The answer is likely between 3 and 8
    
    outer_sides_to_try = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
    
    for os in outer_sides_to_try:
        if time.time() > deadline - 25:
            break
        
        # Try 4 random starts for each outer size
        for trial in range(4):
            if time.time() > deadline - 20:
                break
            
            seed = int(os * 10) * 100 + trial
            params = _random_init(os * 0.5, seed=seed)
            
            opt_params, opt_val = _nm_optimize(
                params, os, outer_center, outer_angle,
                maxiter=80,
                timeout_fn=lambda: time.time() > deadline - 15
            )
            
            # Check validity
            pen = _compute_penalty(opt_params, os, outer_center, outer_angle)
            if pen < 0.01:
                # Valid packing found!
                if best_valid_outer is None or os < best_valid_outer:
                    best_valid_outer = os
                    best_valid_params = opt_params.copy()
            
            if opt_val < best_score:
                best_score = opt_val
                best_params = opt_params.copy()
    
    # If we found a valid packing, try to minimize the outer side further
    if best_valid_outer is not None:
        # Local search on outer side
        lo = max(1.0, best_valid_outer - 1.0)
        hi = best_valid_outer
        
        for _ in range(25):
            if time.time() > deadline - 10:
                break
            mid = (lo + hi) / 2
            pen = _compute_penalty(best_valid_params, mid, outer_center, outer_angle)
            if pen < 0.01:
                hi = mid
                best_valid_outer = mid
            else:
                lo = mid
        
        # Also try to optimize inner positions at the reduced outer size
        if time.time() < deadline - 15:
            opt_params, opt_val = _nm_optimize(
                best_valid_params, best_valid_outer, outer_center, outer_angle,
                maxiter=100,
                timeout_fn=lambda: time.time() > deadline - 10
            )
            pen = _compute_penalty(opt_params, best_valid_outer, outer_center, outer_angle)
            if pen < 0.01:
                best_valid_params = opt_params
                best_valid_outer = best_valid_outer
            else:
                # Try slightly larger
                for delta in [0.1, 0.2, 0.3, 0.5]:
                    test_side = best_valid_outer + delta
                    pen2 = _compute_penalty(opt_params, test_side, outer_center, outer_angle)
                    if pen2 < 0.01:
                        best_valid_outer = test_side
                        best_valid_params = opt_params
                        break
    
    # Final result
    if best_valid_outer is not None:
        final_side = best_valid_outer
        final_params = best_valid_params
    else:
        # Fallback: use best score params with a large outer hex
        final_side = max(best_score, 8.0)
        final_params = best_params
    
    # Build inner list
    inner = []
    for i in range(N):
        x = float(final_params[3*i])
        y = float(final_params[3*i+1])
        ang = float(final_params[3*i+2])
        inner.append([x, y, ang])
    
    return inner, (0.0, 0.0), float(final_side), 0.0

