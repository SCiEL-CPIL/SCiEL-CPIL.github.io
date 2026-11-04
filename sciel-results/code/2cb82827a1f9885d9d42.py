"""Pack 12 unit regular hexagons inside a regular hexagon, minimizing outer side length."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12
DEADLINE = 145.0

def hex_vertices(cx, cy, side, angle_deg):
    angle = math.radians(angle_deg)
    angles = np.array([0, 60, 120, 180, 240, 300], dtype=float) * math.pi / 180.0 + angle
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    corners = hex_vertices(cx, cy, side, angle_deg)
    for i in range(6):
        x1, y1 = corners[i]
        x2, y2 = corners[(i + 1) % 6]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        nx, ny = cx - mx, cy - my
        nl = math.sqrt(nx * nx + ny * ny)
        if nl < 1e-15:
            continue
        nx, ny = nx / nl, ny / nl
        d = (px - x1) * nx + (py - y1) * ny
        if d < -1e-12:
            return False
    return True

def hex_hex_separation(c1, s1, a1, c2, s2, a2):
    """Min separation between two hexagons. Negative if overlapping."""
    corners1 = hex_vertices(c1[0], c1[1], s1, a1)
    corners2 = hex_vertices(c2[0], c2[1], s2, a2)
    min_sep = float('inf')
    # Check all vertex-edge pairs
    for i in range(6):
        for j in range(6):
            px, py = corners1[i]
            if point_in_hexagon(px, py, c2[0], c2[1], s2, a2):
                # Estimate penetration
                max_pen = 0
                for k in range(6):
                    x1, y1 = corners2[k]
                    x2, y2 = corners2[(k + 1) % 6]
                    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                    nx, ny = c2[0] - mx, c2[1] - my
                    nl = math.sqrt(nx * nx + ny * ny)
                    if nl < 1e-15:
                        continue
                    nx, ny = nx / nl, ny / nl
                    d = (px - x1) * nx + (py - y1) * ny
                    if d < max_pen:
                        max_pen = d
                min_sep = min(min_sep, max_pen)
                break
        else:
            continue
        break
    else:
        # No vertex of hex1 inside hex2, check distances
        for i in range(6):
            px, py = corners1[i]
            for j in range(6):
                x1, y1 = corners2[j]
                x2, y2 = corners2[(j + 1) % 6]
                dx, dy = x2 - x1, y2 - y1
                len2 = dx * dx + dy * dy
                if len2 < 1e-15:
                    dist = math.sqrt((px - x1)**2 + (py - y1)**2)
                else:
                    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / len2))
                    qx, qy = x1 + t * dx, y1 + t * dy
                    dist = math.sqrt((px - qx)**2 + (py - qy)**2)
                min_sep = min(min_sep, dist)
    
    # Symmetric check
    for i in range(6):
        for j in range(6):
            px, py = corners2[i]
            if point_in_hexagon(px, py, c1[0], c1[1], s1, a1):
                max_pen = 0
                for k in range(6):
                    x1, y1 = corners1[k]
                    x2, y2 = corners1[(k + 1) % 6]
                    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                    nx, ny = c1[0] - mx, c1[1] - my
                    nl = math.sqrt(nx * nx + ny * ny)
                    if nl < 1e-15:
                        continue
                    nx, ny = nx / nl, ny / nl
                    d = (px - x1) * nx + (py - y1) * ny
                    if d < max_pen:
                        max_pen = d
                min_sep = min(min_sep, max_pen)
                break
        else:
            continue
        break
    
    return min_sep if min_sep != float('inf') else 0.0

def required_outer_side(inner_hexes, ocx=0.0, ocy=0.0, oang=0.0):
    angle = math.radians(oang)
    normal_angles = np.array([30, 90, 150, 210, 270, 330], dtype=float) * math.pi / 180.0 + angle
    max_required = 0
    for na in normal_angles:
        nx, ny = math.cos(na), math.sin(na)
        for (ix, iy, iang) in inner_hexes:
            corners = hex_vertices(ix, iy, 1.0, iang)
            projections = corners[:, 0] * nx + corners[:, 1] * ny
            center_proj = ocx * nx + ocy * ny
            max_proj = np.max(projections) - center_proj
            if max_proj > max_required:
                max_required = max_proj
    return max(0.0, max_required * 2.0 / math.sqrt(3.0))

def construct():
    start_time = time.time()
    
    # Initial hexagonal cluster arrangement
    initial_inner = []
    initial_inner.append([0.0, 0.0, 0.0])
    for k in range(6):
        ang = k * 60
        x = math.sqrt(3) * math.cos(math.radians(ang))
        y = math.sqrt(3) * math.sin(math.radians(ang))
        initial_inner.append([x, y, 0.0])
    for k in range(5):
        ang = 30 + k * 60
        x = 2 * math.sqrt(3) * math.cos(math.radians(ang))
        y = 2 * math.sqrt(3) * math.sin(math.radians(ang))
        initial_inner.append([x, y, 0.0])
    
    def opt_objective(params):
        inner = [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(N)]
        side = required_outer_side(inner)
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                sep = hex_hex_separation(
                    (inner[i][0], inner[i][1]), 1.0, inner[i][2],
                    (inner[j][0], inner[j][1]), 1.0, inner[j][2]
                )
                if sep < 0:
                    penalty += (-sep) ** 2
        return side + 50.0 * penalty
    
    x0 = []
    for (x, y, a) in initial_inner:
        x0.extend([x, y, a])
    
    best_result = np.array(x0)
    best_score = opt_objective(best_result)
    
    elapsed = time.time() - start_time
    if elapsed < 80:
        try:
            de_result = differential_evolution(
                opt_objective, [(-8, 8)] * (3 * N),
                maxiter=30, popsize=8, seed=42, polish=False, init='sobol'
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_result = de_result.x
        except Exception:
            pass
    
    elapsed = time.time() - start_time
    if elapsed < 130:
        try:
            nm_result = minimize(
                opt_objective, best_result, method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9}
            )
            if nm_result.fun < best_score:
                best_score = nm_result.fun
                best_result = nm_result.x
        except Exception:
            pass
    
    inner = [[best_result[3*i], best_result[3*i+1], best_result[3*i+2]] for i in range(N)]
    outer_side = required_outer_side(inner)
    
    return inner, (0.0, 0.0), outer_side, 0.0
