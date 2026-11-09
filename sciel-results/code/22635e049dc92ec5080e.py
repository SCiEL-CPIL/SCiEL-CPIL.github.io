"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def _hex_vertex(cx, cy, side, angle_deg):
    angle_rad = np.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle_rad + i * np.pi / 3
        verts.append([cx + side * np.cos(a), cy + side * np.sin(a)])
    return np.array(verts)

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    dx = px - cx
    dy = py - cy
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    s = side
    sqrt3 = np.sqrt(3)
    if abs(lx) > s:
        return False
    if abs(ly) > s * sqrt3 / 2:
        return False
    if abs(lx / 2 + ly * sqrt3 / 2) > s:
        return False
    if abs(-lx / 2 - ly * sqrt3 / 2) > s:
        return False
    if abs(lx / 2 - ly * sqrt3 / 2) > s:
        return False
    if abs(-lx / 2 + ly * sqrt3 / 2) > s:
        return False
    return True

def _hexagon_overlap(h1, h2):
    cx1, cy1, s1, a1 = h1
    cx2, cy2, s2, a2 = h2
    v1 = _hex_vertex(cx1, cy1, s1, a1)
    v2 = _hex_vertex(cx2, cy2, s2, a2)
    for v in v1:
        if _point_in_hexagon(v[0], v[1], cx2, cy2, s2, a2):
            return True
    for v in v2:
        if _point_in_hexagon(v[0], v[1], cx1, cy1, s1, a1):
            return True
    return False

def _enclosing_side(inner_positions, outer_center, outer_angle):
    cx, cy = outer_center
    oa = outer_angle
    max_s = 0.0
    sqrt3 = np.sqrt(3)
    for ix, iy, ia in inner_positions:
        inner_verts = _hex_vertex(ix, iy, 1.0, ia)
        for vx, vy in inner_verts:
            dx = vx - cx
            dy = vy - cy
            cos_a = np.cos(-np.radians(oa))
            sin_a = np.sin(-np.radians(oa))
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            s_needed = max(abs(lx), abs(ly) / (sqrt3 / 2),
                          abs(lx / 2 + ly * sqrt3 / 2),
                          abs(-lx / 2 - ly * sqrt3 / 2),
                          abs(lx / 2 - ly * sqrt3 / 2),
                          abs(-lx / 2 + ly * sqrt3 / 2))
            if s_needed > max_s:
                max_s = s_needed
    return max_s

def _no_overlap(inner_positions):
    for i in range(len(inner_positions)):
        for j in range(i + 1, len(inner_positions)):
            h1 = (inner_positions[i][0], inner_positions[i][1], 1.0, inner_positions[i][2])
            h2 = (inner_positions[j][0], inner_positions[j][1], 1.0, inner_positions[j][2])
            if _hexagon_overlap(h1, h2):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    sqrt3 = np.sqrt(3)
    
    # Hexagonal cluster: 1 center + 6 ring1 + 4 from ring2
    positions = [[0.0, 0.0, 0.0]]
    for k in range(6):
        angle = k * 60.0
        x = sqrt3 * np.cos(np.radians(angle))
        y = 1.5 * np.sin(np.radians(angle))
        positions.append([x, y, 0.0])
    for k in [0, 60, 120, 180]:
        angle = k
        x = 2 * sqrt3 * np.cos(np.radians(angle))
        y = 3.0 * np.sin(np.radians(angle))
        positions.append([x, y, 0.0])
    
    if not _no_overlap(positions):
        positions = [[k * 3.5, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    
    best_side = _enclosing_side(positions, (0.0, 0.0), 0.0)
    best_oc = (0.0, 0.0)
    best_oa = 0.0
    
    for oa in [0.0, 15.0, 30.0, 45.0, 60.0]:
        for dx in [-0.5, 0.0, 0.5]:
            for dy in [-0.5, 0.0, 0.5]:
                s = _enclosing_side(positions, (dx, dy), oa)
                if s < best_side:
                    best_side = s
                    best_oc = (dx, dy)
                    best_oa = oa
    
    if time.time() < deadline - 30:
        def objective(params):
            if time.time() > deadline:
                return best_side
            inner = [[params[i * 3], params[i * 3 + 1], params[i * 3 + 2]] for i in range(N)]
            if not _no_overlap(inner):
                return 100.0
            return _enclosing_side(inner, best_oc, best_oa)
        
        x0 = np.array([p for pos in positions for p in pos])
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                            options={'maxiter': 300, 'xatol': 0.01, 'fatol': 0.01})
            if result.fun < best_side:
                new_pos = [[result.x[i * 3], result.x[i * 3 + 1], result.x[i * 3 + 2]] for i in range(N)]
                if _no_overlap(new_pos):
                    best_side = result.fun
                    positions = new_pos
        except:
            pass
    
    if not _no_overlap(positions):
        positions = [[k * 3.5, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        best_side = _enclosing_side(positions, (0.0, 0.0), 0.0)
        best_oc = (0.0, 0.0)
        best_oa = 0.0
    
    return positions, best_oc, best_side, best_oa
# EVOLVE-BLOCK-END
