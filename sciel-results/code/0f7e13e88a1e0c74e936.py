"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Direct optimization of all 12 hexagon positions with SLSQP, multiple topologies.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 145.0

# Precomputed unit hexagon data
_ANGLES = np.array([math.radians(a) for a in range(0, 360, 60)])
_UNIT_VERTS_0 = np.column_stack([np.cos(_ANGLES), np.sin(_ANGLES)])
_NORMAL_ANGLES = _ANGLES + math.pi / 6.0
_UNIT_NORMALS_0 = np.column_stack([np.cos(_NORMAL_ANGLES), np.sin(_NORMAL_ANGLES)])


def hex_vertices(cx, cy, s, angle_deg):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    rot = np.array([[ca, -sa], [sa, ca]])
    return (rot @ _UNIT_VERTS_0.T).T * s + np.array([cx, cy])


def compute_min_outer_side(inner, outer_center, outer_angle):
    """Minimum outer side length for given inner hexagons."""
    cx, cy = outer_center
    a = math.radians(outer_angle)
    ca, sa = math.cos(a), math.sin(a)
    rot = np.array([[ca, -sa], [sa, ca]])
    normals = (rot @ _UNIT_NORMALS_0.T).T

    max_proj = 0.0
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        rel = p - np.array([cx, cy])
        projs = rel @ normals.T
        max_proj = max(max_proj, projs.max())
    return (2.0 / math.sqrt(3.0)) * max_proj


def sat_gap(A, B):
    """Maximum SAT gap between two polygons. Positive = separated."""
    max_gap = -float('inf')
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            norm = math.sqrt(dx * dx + dy * dy)
            if norm < 1e-9:
                continue
            nx, ny = -dy / norm, dx / norm
            projA = A @ np.array([nx, ny])
            projB = B @ np.array([nx, ny])
            minA, maxA = projA.min(), projA.max()
            minB, maxB = projB.min(), projB.max()
            if maxA < minB:
                gap = minB - maxA
            elif maxB < minA:
                gap = minA - maxB
            else:
                gap = -(min(maxA, maxB) - max(minA, minB))
            if gap > max_gap:
                max_gap = gap
    return max_gap


def check_validity(inner, outer_center, outer_side, outer_angle, eps=1e-7):
    """Check all inner hexagons are inside outer and pairwise non-overlapping."""
    outer = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    polys = [hex_vertices(x, y, 1.0, ang) for (x, y, ang) in inner]
    
    # Check containment
    outer_list = [(outer[j, 0], outer[j, 1]) for j in range(6)]
    for p in polys:
        for i in range(6):
            px, py = p[i, 0], p[i, 1]
            sign = 0
            for m in range(6):
                x1, y1 = outer_list[m]
                x2, y2 = outer_list[(m + 1) % 6]
                cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
                if abs(cross) < 1e-9:
                    continue
                s = 1 if cross > 0 else -1
                if sign == 0:
                    sign = s
                elif s != sign:
                    return False
    # Check pairwise non-overlap
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if sat_gap(polys[i], polys[j]) < -eps:
                return False
    return True


def objective_direct(params, outer_angle, k_smooth=20.0):
    """
    Direct optimization: params = 24 values (12 x, 12 y), all hexagons have angle 0.
    Returns smooth objective with soft constraints.
    """
    xs = params[:12]
    ys = params[12:]
    inner = [(xs[i], ys[i], 0.0) for i in range(12)]
    
    cx, cy = 0.0, 0.0
    a = math.radians(outer_angle)
    ca, sa = math.cos(a), math.sin(a)
    rot = np.array([[ca, -sa], [sa, ca]])
    normals = (rot @ _UNIT_NORMALS_0.T).T
    
    # Compute max projection for outer side
    max_proj = 0.0
    polys = []
    for i in range(12):
        p = hex_vertices(xs[i], ys[i], 1.0, 0.0)
        polys.append(p)
        rel = p - np.array([cx, cy])
        projs = rel @ normals.T
        max_proj = max(max_proj, projs.max())
    
    side = (2.0 / math.sqrt(3.0)) * max_proj
    
    # Overlap penalty
    overlap_penalty = 0.0
    for i in range(12):
        for j in range(i + 1, 12):
            gap = sat_gap(polys[i], polys[j])
            if gap < 0:
                # Soft penalty: use smooth max approximation
                overlap_penalty += (-gap) ** 2
    
    return side + 50.0 * overlap_penalty


def objective_direct_hard(params, outer_angle):
    """Hard version: returns side if valid, else side + penalty."""
    xs = params[:12]
    ys = params[12:]
    inner = [(xs[i], ys[i], 0.0) for i in range(12)]
    
    side = compute_min_outer_side(inner, (0.0, 0.0), outer_angle)
    
    if not check_validity(inner, (0.0, 0.0), side, outer_angle):
        # Compute violation magnitude
        polys = [hex_vertices(xs[i], ys[i], 1.0, 0.0) for i in range(12)]
        max_violation = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                gap = sat_gap(polys[i], polys[j])
                if gap < 0:
                    max_violation = max(max_violation, -gap)
        return side + 100.0 * max_violation + 10.0
    return side


def build_hexagonal_ring(scale, outer_angle):
    """12 hexagons in a ring around center, all with angle 0."""
    inner = []
    for k in range(12):
        theta = math.radians(k * 30.0)
        x = scale * math.cos(theta)
        y = scale * math.sin(theta)
        inner.append((x, y, 0.0))
    return inner


def build_two_rings(r1, r2, outer_angle):
    """6 in inner ring + 6 in outer ring."""
    inner = []
    for k in range(6):
        theta = math.radians(k * 60.0)
        x = r1 * math.cos(theta)
        y = r1 * math.sin(theta)
        inner.append((x, y, 0.0))
    for k in range(6):
        theta = math.radians(k * 60.0 + 30.0)
        x = r2 * math.cos(theta)
        y = r2 * math.sin(theta)
        inner.append((x, y, 0.0))
    return inner


def build_hex_lattice_flat(spacing_x, spacing_y, outer_angle):
    """3x4 grid of hexagons."""
    inner = []
    for i in range(3):
        for j in range(4):
            x = (i - 1) * spacing_x
            y = (j - 1.5) * spacing_y
            inner.append((x, y, 0.0))
    return inner


def optimize_direct(outer_angle, time_budget, init_params=None):
    """Optimize 24 parameters (12 x, 12 y) directly."""
    start_time = time.time()
    
    if init_params is None:
        # Initialize with hexagonal ring
        scale = 2.0
        inner = build_hexagonal_ring(scale, outer_angle)
        init_params = []
        for (x, y, a) in inner:
            init_params.extend([x, y])
    
    x0 = np.array(init_params, dtype=float)
    
    # Bounds
    bounds = [(-5.0, 5.0)] * 24
    
    best_x = x0.copy()
    best_f = objective_direct_hard(x0, outer_angle)
    
    # SLSQP optimization
    if time.time() - start_time < time_budget * 0.6:
        try:
            result = minimize(objective_direct, x0, args=(outer_angle,),
                              method='SLSQP', bounds=bounds,
                              options={'maxiter': 500, 'ftol': 1e-6})
            if result.fun < best_f:
                exact_f = objective_direct_hard(result.x, outer_angle)
                if exact_f < best_f:
                    best_x = result.x.copy()
                    best_f = exact_f
        except Exception:
            pass
    
    # Nelder-Mead refinement
    if time.time() - start_time < time_budget * 0.8:
        try:
            result = minimize(objective_direct_hard, best_x, args=(outer_angle,),
                              method='Nelder-Mead', bounds=bounds,
                              options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < best_f:
                best_x = result.x.copy()
                best_f = result.fun
        except Exception:
            pass
    
    # High-precision polish
    if time.time() - start_time < time_budget * 0.95:
        try:
            result = minimize(objective_direct_hard, best_x, args=(outer_angle,),
                              method='Nelder-Mead', bounds=bounds,
                              options={'maxiter': 5000, 'xatol': 1e-11, 'fatol': 1e-13})
            if result.fun < best_f:
                best_x = result.x.copy()
                best_f = result.fun
        except Exception:
            pass
    
    return best_x, best_f


def optimize_with_rotation(outer_angle, time_budget):
    """Try multiple initial configurations and optimize."""
    start_time = time.time()
    best_x = None
    best_f = float('inf')
    
    # Try different initial configurations
    configs = []
    
    # Hexagonal ring
    for scale in [1.8, 2.0, 2.2]:
        inner = build_hexagonal_ring(scale, outer_angle)
        params = []
        for (x, y, a) in inner:
            params.extend([x, y])
        configs.append(params)
    
    # Two rings
    for r1, r2 in [(1.5, 3.0), (1.8, 3.2), (2.0, 3.5)]:
        inner = build_two_rings(r1, r2, outer_angle)
        params = []
        for (x, y, a) in inner:
            params.extend([x, y])
        configs.append(params)
    
    # Grid
    for sx, sy in [(1.7, 1.7), (1.8, 1.8), (2.0, 2.0)]:
        inner = build_hex_lattice_flat(sx, sy, outer_angle)
        params = []
        for (x, y, a) in inner:
            params.extend([x, y])
        configs.append(params)
    
    # Optimize each config
    for params in configs:
        if time.time() > DEADLINE - 10.0:
            break
        time_left = min(time_budget, DEADLINE - time.time() - 10.0)
        if time_left < 2.0:
            break
        x_opt, f_opt = optimize_direct(outer_angle, time_left, init_params=params)
        if f_opt < best_f:
            best_f = f_opt
            best_x = x_opt
    
    return best_x, best_f


def construct():
    t0 = time.time()
    
    best_side = 40.0
    best_inner = None
    best_outer_angle = 0.0
    best_valid = False
    
    outer_angles = [0.0, 30.0, 60.0]
    
    total_time = 138.0
    time_per_angle = total_time / len(outer_angles) * 0.85
    
    for orot in outer_angles:
        if time.time() > DEADLINE - 15.0:
            break
        time_budget = min(time_per_angle, DEADLINE - time.time() - 15.0)
        if time_budget < 5.0:
            break
        
        x_opt, f_opt = optimize_with_rotation(orot, time_budget)
        
        if x_opt is not None:
            xs = x_opt[:12]
            ys = x_opt[12:]
            inner = [(xs[i], ys[i], 0.0) for i in range(12)]
            side = compute_min_outer_side(inner, (0.0, 0.0), orot)
            
            if check_validity(inner, (0.0, 0.0), side, orot):
                if side < best_side:
                    best_side = side
                    best_inner = inner
                    best_outer_angle = orot
                    best_valid = True
    
    if best_valid and best_inner is not None:
        return best_inner, (0.0, 0.0), best_side, best_outer_angle
    
    # Fallback: linear arrangement
    inner = [(4.0 * k, 0.0, 0.0) for k in range(-6, 6)]
    return inner, (0.0, 0.0), 40.0, 0.0
