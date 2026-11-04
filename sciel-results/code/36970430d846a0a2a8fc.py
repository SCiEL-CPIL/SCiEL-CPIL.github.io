"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Direct per-hexagon optimization with smooth epigraph formulation.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 145.0

# --- Geometry helpers ---

def hex_vertices(cx, cy, s, angle_deg):
    a = math.radians(angle_deg)
    return [(cx + s * math.cos(a + i * math.pi / 3.0),
             cy + s * math.sin(a + i * math.pi / 3.0)) for i in range(6)]

def hex_vertices_np(cx, cy, s, angle_rad):
    """Vectorized vertices as (6,2) array."""
    angles = angle_rad + np.arange(6) * (math.pi / 3.0)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def point_in_convex_polygon(px, py, poly):
    n = len(poly)
    sign = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) < 1e-9:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True

def polygons_overlap_sat(A, B, eps=1e-9):
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            if nx == 0 and ny == 0:
                continue
            minA = min(nx * px + ny * py for px, py in A)
            maxA = max(nx * px + ny * py for px, py in A)
            minB = min(nx * px + ny * py for px, py in B)
            maxB = max(nx * px + ny * py for px, py in B)
            if maxA < minB - eps or maxB < minA - eps:
                return False
    return True

def check_validity(inner, outer_center, outer_side, outer_angle, eps=1e-7):
    outer = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    polys = [hex_vertices(x, y, 1.0, ang) for (x, y, ang) in inner]
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                return False
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap_sat(polys[i], polys[j]):
                return False
    return True

def compute_min_outer_side(inner, outer_center, outer_angle):
    cx, cy = outer_center
    a = math.radians(outer_angle)
    normals = np.column_stack([
        np.cos(a + (i + 0.5) * math.pi / 3.0),
        np.sin(a + (i + 0.5) * math.pi / 3.0)
    ] for i in range(6))
    max_proj = 0.0
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        for (px, py) in p:
            dx, dy = px - cx, py - cy
            for (nx, ny) in normals:
                proj = dx * nx + dy * ny
                if proj > max_proj:
                    max_proj = proj
    return (2.0 / math.sqrt(3.0)) * max_proj

# --- Smooth helpers ---

def smooth_max(values, k=50.0):
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return 0.0
    v_max = float(np.max(v))
    if v_max < -1e10:
        return v_max
    shifted = np.clip(v - v_max, -50.0, 50.0)
    return v_max + math.log(np.sum(np.exp(k * shifted)) / k) / k

def sat_gap_np(A, B):
    """Maximum SAT gap between two (6,2) vertex arrays. Positive=separated."""
    max_gap = -1e10
    for poly in (A, B):
        n = 6
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            norm = math.sqrt(nx*nx + ny*ny)
            if norm < 1e-9:
                continue
            nx, ny = nx/norm, ny/norm
            projA = nx * A[:, 0] + ny * A[:, 1]
            projB = nx * B[:, 0] + ny * B[:, 1]
            minA, maxA = float(np.min(projA)), float(np.max(projA))
            minB, maxB = float(np.min(projB)), float(np.max(projB))
            if maxA < minB:
                gap = minB - maxA
            elif maxB < minA:
                gap = minA - maxB
            else:
                gap = -(min(maxA, maxB) - max(minA, minB))
            if gap > max_gap:
                max_gap = gap
    return max_gap

# --- Objective functions ---

def objective_direct(params, outer_rot, k_smooth=30.0):
    """
    Direct optimization: params = [x0,y0,a0, x1,y1,a1, ..., x11,y11,a11]
    Outer center fixed at origin. Returns smooth outer side + overlap penalty.
    """
    n_params = len(params)
    if n_params != 36:
        return 1000.0
    
    inner = []
    verts_list = []
    for i in range(12):
        x = params[3*i]
        y = params[3*i+1]
        ang_deg = params[3*i+2]
        inner.append((x, y, ang_deg))
        verts_list.append(hex_vertices_np(x, y, 1.0, math.radians(ang_deg)))
    
    # Compute smooth outer side
    a = math.radians(outer_rot)
    normals = np.column_stack([
        np.cos(a + (i + 0.5) * math.pi / 3.0),
        np.sin(a + (i + 0.5) * math.pi / 3.0)
    ] for i in range(6))
    
    all_projs = []
    for verts in verts_list:
        for (px, py) in verts:
            for (nx, ny) in normals:
                all_projs.append(px * nx + py * ny)
    
    side_smooth = (2.0 / math.sqrt(3.0)) * smooth_max(all_projs, k_smooth)
    
    # Overlap penalty (smooth)
    overlap_penalty = 0.0
    for i in range(12):
        for j in range(i + 1, 12):
            gap = sat_gap_np(verts_list[i], verts_list[j])
            if gap < 0:
                overlap_penalty += (-gap) ** 2
    
    return side_smooth + 5.0 * overlap_penalty

def objective_direct_hard(params, outer_rot):
    """Hard validity check version."""
    inner = []
    for i in range(12):
        x = params[3*i]
        y = params[3*i+1]
        ang_deg = params[3*i+2]
        inner.append((x, y, ang_deg))
    
    side = compute_min_outer_side(inner, (0.0, 0.0), outer_rot)
    if not check_validity(inner, (0.0, 0.0), side, outer_rot):
        # Penalty
        polys = [hex_vertices(x, y, 1.0, ang) for (x, y, ang) in inner]
        max_viol = 0.0
        outer = hex_vertices(0.0, 0.0, side, outer_rot)
        for p in polys:
            for (px, py) in p:
                for i in range(6):
                    x1, y1 = outer[i]
                    x2, y2 = outer[(i+1)%6]
                    cross = (x2-x1)*(py-y1) - (y2-y1)*(px-x1)
                    if cross < -1e-7:
                        max_viol = max(max_viol, -cross)
        for i in range(12):
            for j in range(i+1, 12):
                gap = sat_gap_np(np.array(polys[i]), np.array(polys[j]))
                if gap < 0:
                    max_viol = max(max_viol, -gap)
        return side + 10.0 * max_viol + 5.0
    return side

# --- Initial configurations ---

def make_ring12(scale, inner_rot):
    """12 hexagons in a ring."""
    params = np.zeros(36)
    for k in range(12):
        theta = math.radians(inner_rot + k * 30.0)
        params[3*k] = scale * math.cos(theta)
        params[3*k+1] = scale * math.sin(theta)
        params[3*k+2] = inner_rot
    return params

def make_hex_lattice(r1, r2):
    """1 center + 6 at r1 + 5 at r2."""
    params = np.zeros(36)
    # Center
    params[0] = 0.0; params[1] = 0.0; params[2] = 0.0
    # Ring 1: 6 hexagons
    for k in range(6):
        theta = math.radians(k * 60.0)
        params[3*(1+k)] = r1 * math.cos(theta)
        params[3*(1+k)+1] = r1 * math.sin(theta)
        params[3*(1+k)+2] = 0.0
    # Ring 2: 5 hexagons
    for k in range(5):
        theta = math.radians(k * 72.0 + 18.0)
        params[3*(7+k)] = r2 * math.cos(theta)
        params[3*(7+k)+1] = r2 * math.sin(theta)
        params[3*(7+k)+2] = 0.0
    return params

def make_rect3x4(sx, sy, rot):
    """3x4 grid."""
    params = np.zeros(36)
    idx = 0
    for i in range(3):
        for j in range(4):
            x = (i - 1) * sx
            y = (j - 1.5) * sy
            # Apply rotation
            a = math.radians(rot)
            xr = x * math.cos(a) - y * math.sin(a)
            yr = x * math.sin(a) + y * math.cos(a)
            params[3*idx] = xr
            params[3*idx+1] = yr
            params[3*idx+2] = rot
            idx += 1
    return params

def make_linear(spacing):
    """12 in a line."""
    params = np.zeros(36)
    for k in range(12):
        params[3*k] = (k - 5.5) * spacing
        params[3*k+1] = 0.0
        params[3*k+2] = 0.0
    return params

def make_hex6_plus6(scale1, scale2, rot1, rot2):
    """Two rings of 6."""
    params = np.zeros(36)
    for k in range(6):
        theta = math.radians(rot1 + k * 60.0)
        params[3*k] = scale1 * math.cos(theta)
        params[3*k+1] = scale1 * math.sin(theta)
        params[3*k+2] = rot1
    for k in range(6):
        theta = math.radians(rot2 + k * 60.0)
        params[3*(6+k)] = scale2 * math.cos(theta)
        params[3*(6+k)+1] = scale2 * math.sin(theta)
        params[3*(6+k)+2] = rot2
    return params

# --- Optimization ---

def optimize_from_start(start_params, outer_rot, time_budget):
    """Run optimization from a given starting point."""
    t_start = time.time()
    best_x = start_params.copy()
    best_f = objective_direct_hard(best_x, outer_rot)
    
    # Phase 1: SLSQP with smooth objective
    if time.time() - t_start < time_budget * 0.4:
        try:
            bounds = [(-10, 10), (-10, 10), (-180, 180)] * 12
            result = minimize(objective_direct, best_x, args=(outer_rot, 30.0),
                             method='SLSQP', bounds=bounds,
                             options={'maxiter': 200, 'ftol': 1e-7})
            if result.fun < best_f:
                cand = result.x
                hard_f = objective_direct_hard(cand, outer_rot)
                if hard_f < best_f:
                    best_x = cand
                    best_f = hard_f
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead with hard objective
    if time.time() - t_start < time_budget * 0.75:
        try:
            bounds = [(-10, 10), (-10, 10), (-180, 180)] * 12
            result = minimize(objective_direct_hard, best_x, args=(outer_rot),
                             method='Nelder-Mead',
                             options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9})
            if result.fun < best_f:
                best_x = result.x
                best_f = result.fun
        except Exception:
            pass
    
    # Phase 3: Fine polish
    if time.time() - t_start < time_budget * 0.95:
        try:
            result = minimize(objective_direct_hard, best_x, args=(outer_rot),
                             method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11})
            if result.fun < best_f:
                best_x = result.x
                best_f = result.fun
        except Exception:
            pass
    
    return best_x, best_f

def optimize():
    best_side = 40.0
    best_params = None
    best_outer_rot = 0.0
    best_valid = False
    
    outer_rots = [0.0, 30.0, 60.0]
    
    # Multiple initial configurations
    starts = [
        ("ring12", make_ring12(2.0, 0.0)),
        ("ring12b", make_ring12(2.2, 15.0)),
        ("lattice", make_hex_lattice(1.73, 3.0)),
        ("lattice2", make_hex_lattice(1.8, 3.2)),
        ("rect", make_rect3x4(2.0, 2.0, 0.0)),
        ("rect2", make_rect3x4(2.1, 1.9, 30.0)),
        ("linear", make_linear(1.8)),
        ("hex6+6", make_hex6_plus6(1.5, 3.0, 0.0, 30.0)),
        ("hex6+6b", make_hex6_plus6(1.8, 3.2, 15.0, 45.0)),
    ]
    
    time_per = 12.0
    
    for orot in outer_rots:
        for name, start in starts:
            if time.time() > DEADLINE:
                break
            params, f = optimize_from_start(start, orot, time_per)
            if f < best_side:
                # Verify
                inner = [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(12)]
                side = compute_min_outer_side(inner, (0.0, 0.0), orot)
                if check_validity(inner, (0.0, 0.0), side, orot):
                    if side < best_side:
                        best_side = side
                        best_params = params.copy()
                        best_outer_rot = orot
                        best_valid = True
    
    return best_side, best_params, best_outer_rot, best_valid

def construct():
    t0 = time.time()
    try:
        side, params, outer_rot, valid = optimize()
        if valid and params is not None:
            inner = [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(12)]
            return inner, (0.0, 0.0), side, outer_rot
    except Exception:
        pass
    
    # Fallback: ring of 12
    inner = []
    for k in range(12):
        theta = math.radians(k * 30.0)
        inner.append((2.5 * math.cos(theta), 2.5 * math.sin(theta), 0.0))
    return inner, (0.0, 0.0), 5.0, 0.0
