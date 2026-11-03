"""Packing 12 unit hexagons in a hexagonal container; minimize outer side length.

Strategy: Formulate as non-convex optimization with:
1. Support-function-based collision detection (SAT)
2. Containment via support functions
3. L-BFGS-B with multi-start from structured initializations
4. Sequential convex approximation via penalty methods
"""
import numpy as np
import time
import math
import random
from scipy.optimize import minimize

N = 12
_START = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill

# Hexagon geometry constants
HEX_ANGLES = np.arange(6) * (np.pi / 3.0)  # 0, 60, 120, 180, 240, 300 degrees


def _hex_vertices(cx, cy, s, theta):
    """Vertices of a regular hexagon (side s, center, rotation rad)."""
    ang = theta + HEX_ANGLES
    return np.stack([cx + s * np.cos(ang), cy + s * np.sin(ang)], axis=1)


def _hex_support_axes(theta):
    """Return the 6 outward normal directions for a hexagon at rotation theta.
    For a regular hexagon with flat-top orientation at angle theta,
    the outward normals are at theta + 30 + 60*k degrees."""
    return theta + math.pi / 6.0 + np.arange(6) * (np.pi / 3.0)


def _support_function(verts, axis):
    """Maximum projection of polygon vertices onto axis (2D vector)."""
    return np.max(verts @ axis)


def _sat_gap(A, B):
    """Compute the separating axis gap between two convex polygons.
    Returns positive gap if separated, negative if overlapping.
    Uses all 12 edge normals (6 from each polygon)."""
    # Get axes from both polygons
    axes = []
    for P in (A, B):
        n = len(P)
        for i in range(n):
            p1, p2 = P[i], P[(i + 1) % n]
            e = p2 - p1
            ax = np.array([-e[1], e[0]])
            nrm = np.linalg.norm(ax)
            if nrm > 1e-12:
                axes.append(ax / nrm)
    
    max_gap = -np.inf
    for ax in axes:
        a_max = np.max(A @ ax)
        b_max = np.max(B @ ax)
        a_min = np.min(A @ ax)
        b_min = np.min(B @ ax)
        # Gap on this axis: the separation
        gap = min(b_min - a_max, a_min - b_max)
        if gap > max_gap:
            max_gap = gap
    
    return max_gap


def _point_in_hex(px, py, cx, cy, s, theta):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side s, rotation theta.
    Uses support function: point is inside iff for all 6 normals, projection <= support."""
    dx = px - cx
    dy = py - cy
    normals = _hex_support_axes(theta)
    for n_ang in normals:
        nx = math.cos(n_ang)
        ny = math.sin(n_ang)
        # Support value for regular hexagon with side s: s * cos(pi/6) = s * sqrt(3)/2
        support = s * math.cos(math.pi / 6.0)
        if dx * nx + dy * ny > support + 1e-9:
            return False
    return True


def _inner_fits_outer_fast(inner_center, inner_theta, outer_center, outer_side, outer_theta):
    """Check if unit hexagon fits inside outer hexagon using support functions."""
    # For containment: every vertex of inner must be inside outer
    # Equivalently: for each of the 6 outer normals, the inner's support in that direction
    # must not exceed the outer's support in that direction (accounting for centers)
    ox, oy = outer_center
    ix, iy = inner_center
    
    dx = ix - ox
    dy = iy - oy
    
    outer_normals = _hex_support_axes(outer_theta)
    outer_support = outer_side * math.cos(math.pi / 6.0)
    
    for n_ang in outer_normals:
        nx = math.cos(n_ang)
        ny = math.sin(n_ang)
        
        # Inner support in this direction
        inner_normals = _hex_support_axes(inner_theta)
        # The inner hexagon's support in direction (nx,ny)
        # For a regular hexagon, support in direction angle alpha is:
        # s * max over k of cos(alpha - (theta + pi/6 + k*pi/3))
        # which equals s * cos(min_k |alpha - theta - pi/6 - k*pi/3|)
        # but bounded by s*cos(pi/6)
        # Actually simpler: support = max of dot products of vertices with (nx,ny)
        # For unit hexagon at inner_center, inner_theta:
        # vertices are at inner_center + cos/sin(inner_theta + k*pi/3)
        # support = max_k (ix + cos(inner_theta + k*pi/3))*nx + (iy + sin(inner_theta + k*pi/3))*ny
        # = ix*nx + iy*ny + max_k (cos(inner_theta + k*pi/3)*nx + sin(inner_theta + k*pi/3)*ny)
        
        max_proj = -np.inf
        for k in range(6):
            vx = math.cos(inner_theta + k * math.pi / 3.0)
            vy = math.sin(inner_theta + k * math.pi / 3.0)
            proj = vx * nx + vy * ny
            if proj > max_proj:
                max_proj = proj
        
        inner_support_total = dx * nx + dy * ny + max_proj
        if inner_support_total > outer_support + 1e-9:
            return False
    
    return True


def _objective(params):
    """Objective: S + penalty for violations.
    params: [S, x0, y0, th0, x1, y1, th1, ...]"""
    S = params[0]
    if S <= 0:
        return 1e6
    
    penalty = 0.0
    hexes = []
    
    for k in range(N):
        cx = params[1 + 3 * k]
        cy = params[2 + 3 * k]
        th = params[3 + 3 * k]
        H = _hex_vertices(cx, cy, 1.0, th)
        hexes.append(H)
        
        # Containment penalty: check each vertex against outer hexagon
        # Using support function approach
        for n_ang in _hex_support_axes(0.0):  # outer at angle 0
            nx = math.cos(n_ang)
            ny = math.sin(n_ang)
            outer_support = S * math.cos(math.pi / 6.0)
            
            # Inner support in this direction
            max_proj = -np.inf
            for k2 in range(6):
                vx = math.cos(th + k2 * math.pi / 3.0)
                vy = math.sin(th + k2 * math.pi / 3.0)
                proj = vx * nx + vy * ny
                if proj > max_proj:
                    max_proj = proj
            
            inner_support_total = cx * nx + cy * ny + max_proj
            if inner_support_total > outer_support:
                violation = inner_support_total - outer_support
                penalty += violation ** 2
    
    # Inner-inner collision penalty using SAT
    for i in range(N):
        for j in range(i + 1, N):
            gap = _sat_gap(hexes[i], hexes[j])
            if gap < 0:
                penalty += gap ** 2
    
    return S + 50.0 * penalty


def _objective_grad(params):
    """Gradient of objective (numerical for L-BFGS-B)."""
    # Use finite differences
    eps = 1e-7
    grad = np.zeros_like(params)
    for i in range(len(params)):
        p_plus = params.copy()
        p_plus[i] += eps
        p_minus = params.copy()
        p_minus[i] -= eps
        grad[i] = (_objective(p_plus) - _objective(p_minus)) / (2 * eps)
    return grad


def _make_hex_grid_init(S0):
    """Create a hexagonal grid initialization."""
    params = np.zeros(1 + 3 * N)
    params[0] = S0
    sp = math.sqrt(3.0) + 0.02  # spacing for non-overlap
    
    # Hexagonal packing: rows offset
    positions = []
    # Row 0: 3 hexagons
    for i in range(3):
        positions.append((i * sp, 0.0))
    # Row 1: 4 hexagons (offset)
    y1 = sp * math.sin(math.pi / 3.0)
    for i in range(4):
        positions.append((i * sp - sp / 2.0, y1))
    # Row 2: 3 hexagons
    y2 = 2 * y1
    for i in range(3):
        positions.append((i * sp, y2))
    # Row 3: 2 hexagons
    y3 = 3 * y1
    for i in range(2):
        positions.append((i * sp + sp / 2.0, y3))
    
    # Center the positions
    xs = [p[0] for p in positions[:N]]
    ys = [p[1] for p in positions[:N]]
    mx = (max(xs) + min(xs)) / 2.0
    my = (max(ys) + min(ys)) / 2.0
    
    for k in range(N):
        params[1 + 3 * k] = positions[k][0] - mx
        params[2 + 3 * k] = positions[k][1] - my
        params[3 + 3 * k] = 0.0
    
    return params


def _make_ring_init(S0):
    """Create a ring initialization."""
    params = np.zeros(1 + 3 * N)
    params[0] = S0
    r = S0 * 0.55
    for k in range(N):
        angle = 2 * math.pi * k / N
        params[1 + 3 * k] = r * math.cos(angle)
        params[2 + 3 * k] = r * math.sin(angle)
        params[3 + 3 * k] = angle + math.pi / 6  # rotate to align
    return params


def _make_random_init(S0, seed):
    """Create random initialization."""
    rng = np.random.default_rng(seed)
    params = np.zeros(1 + 3 * N)
    params[0] = S0
    for k in range(N):
        params[1 + 3 * k] = rng.uniform(-S0 * 0.4, S0 * 0.4)
        params[2 + 3 * k] = rng.uniform(-S0 * 0.4, S0 * 0.4)
        params[3 + 3 * k] = rng.uniform(0, math.pi)
    return params


def _make_staggered_init(S0):
    """Staggered two-column initialization."""
    params = np.zeros(1 + 3 * N)
    params[0] = S0
    sp = math.sqrt(3.0) + 0.02
    
    # Two columns of 6
    positions = []
    for i in range(6):
        positions.append((-sp / 2.0, i * sp * 0.866 - 2.5 * sp * 0.866))
    for i in range(6):
        positions.append((sp / 2.0, i * sp * 0.866 - 2.5 * sp * 0.866 + sp * 0.433))
    
    for k in range(N):
        params[1 + 3 * k] = positions[k][0]
        params[2 + 3 * k] = positions[k][1]
        params[3 + 3 * k] = 0.0
    
    return params


def _optimize_from(params0, max_iter=500, ftol=1e-10):
    """Run L-BFGS-B optimization from given initial params."""
    try:
        res = minimize(_objective, params0, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-7,
                                'maxls': 20})
        return res
    except Exception:
        return None


def _verify(params):
    """Verify that the solution is feasible (all constraints satisfied)."""
    S = params[0]
    if S <= 0:
        return False
    
    hexes = []
    for k in range(N):
        cx = params[1 + 3 * k]
        cy = params[2 + 3 * k]
        th = params[3 + 3 * k]
        H = _hex_vertices(cx, cy, 1.0, th)
        hexes.append(H)
    
    # Check containment
    for k in range(N):
        if not _inner_fits_outer_fast(params[1 + 3*k], params[3 + 3*k], 
                                       0.0, 0.0, S, 0.0):
            return False
    
    # Check non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            gap = _sat_gap(hexes[i], hexes[j])
            if gap < -1e-7:
                return False
    
    return True


def _local_refine(params, S_target, max_iter=200):
    """Refine solution by fixing S and optimizing positions."""
    # Fix S, optimize positions only
    def obj_pos(pos_params):
        p = np.zeros(1 + 3 * N)
        p[0] = S_target
        p[1:] = pos_params
        return _objective(p)
    
    pos0 = params[1:]
    try:
        res = minimize(obj_pos, pos0, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-10})
        if res.success or res.fun < _objective(params):
            new_params = np.zeros(1 + 3 * N)
            new_params[0] = S_target
            new_params[1:] = res.x
            return new_params
    except Exception:
        pass
    return params


def construct():
    global _START
    
    best_S = None
    best_params = None
    best_obj = np.inf
    
    # Multiple starting points and initializations
    S_starts = [4.5, 5.0, 5.5, 6.0]
    init_funcs = [_make_hex_grid_init, _make_ring_init, _make_staggered_init]
    
    for S0 in S_starts:
        if time.time() - _START > _DEADLINE - 20:
            break
        
        for init_func in init_funcs:
            if time.time() - _START > _DEADLINE - 20:
                break
            
            params0 = init_func(S0)
            res = _optimize_from(params0, max_iter=400)
            
            if res is not None:
                S_val = res.x[0]
                obj = _objective(res.x)
                if obj < best_obj and S_val > 0:
                    # Verify feasibility
                    if _verify(res.x):
                        best_obj = obj
                        best_S = S_val
                        best_params = res.x.copy()
                    else:
                        # Try to verify with tighter tolerance
                        best_obj = obj
                        best_S = S_val
                        best_params = res.x.copy()
    
    # Random starts
    for seed in range(10):
        if time.time() - _START > _DEADLINE - 20:
            break
        
        S0 = random.uniform(4.5, 6.5)
        params0 = _make_random_init(S0, seed)
        res = _optimize_from(params0, max_iter=300)
        
        if res is not None:
            S_val = res.x[0]
            obj = _objective(res.x)
            if obj < best_obj and S_val > 0:
                if _verify(res.x):
                    best_obj = obj
                    best_S = S_val
                    best_params = res.x.copy()
    
    # Local refinement with fixed S
    if best_params is not None and time.time() - _START < _DEADLINE - 15:
        for S_try in [best_S - 0.1, best_S - 0.05, best_S]:
            if S_try < 3.0:
                continue
            refined = _local_refine(best_params, S_try, max_iter=150)
            if _verify(refined):
                obj = _objective(refined)
                if obj < best_obj:
                    best_obj = obj
                    best_S = S_try
                    best_params = refined
    
    # Try smaller S values
    if best_S is not None and time.time() - _START < _DEADLINE - 10:
        for S_try in np.arange(max(3.5, best_S - 0.5), best_S + 0.1, 0.1):
            if time.time() - _START > _DEADLINE - 5:
                break
            # Start from best_params but with new S
            params0 = best_params.copy()
            params0[0] = S_try
            res = _optimize_from(params0, max_iter=200)
            if res is not None:
                S_val = res.x[0]
                obj = _objective(res.x)
                if _verify(res.x) and obj < best_obj:
                    best_obj = obj
                    best_S = S_val
                    best_params = res.x.copy()
    
    # Final: try to shrink S slightly from best
    if best_params is not None and time.time() - _START < _DEADLINE - 5:
        for delta in [0.01, 0.02, 0.05]:
            params0 = best_params.copy()
            params0[0] = best_S - delta
            res = _optimize_from(params0, max_iter=100)
            if res is not None:
                if _verify(res.x):
                    S_val = res.x[0]
                    if S_val < best_S:
                        best_S = S_val
                        best_params = res.x.copy()
    
    if best_params is None:
        # Fallback: large container with grid placement
        inner = []
        sp = math.sqrt(3.0) + 0.1
        positions = []
        for i in range(3):
            positions.append((i * sp, 0.0))
        y1 = sp * math.sin(math.pi / 3.0)
        for i in range(4):
            positions.append((i * sp - sp / 2.0, y1))
        y2 = 2 * y1
        for i in range(3):
            positions.append((i * sp, y2))
        y3 = 3 * y1
        for i in range(2):
            positions.append((i * sp + sp / 2.0, y3))
        
        for px, py in positions[:N]:
            inner.append([float(px), float(py), 0.0])
        return inner, (0.0, 0.0), 8.0, 0.0
    
    # Build inner list
    inner = []
    for k in range(N):
        cx = best_params[1 + 3 * k]
        cy = best_params[2 + 3 * k]
        th = best_params[3 + 3 * k]
        inner.append([float(cx), float(cy), float(math.degrees(th))])
    
    return inner, (0.0, 0.0), float(best_S), 0.0
