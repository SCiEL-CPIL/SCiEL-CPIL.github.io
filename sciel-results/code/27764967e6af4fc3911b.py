import numpy as np
import time
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def hex_vertices(cx, cy, side=1.0, angle_deg=0.0):
        angles = np.radians(angle_deg) + np.arange(6) * np.pi / 3
        return np.column_stack([cx + side * np.cos(angles), 
                                cy + side * np.sin(angles)])
    
    def point_in_hexagon(px, py, cx, cy, side=1.0, angle_deg=0.0):
        angle = np.radians(angle_deg)
        dx, dy = px - cx, py - cy
        rx = dx * np.cos(-angle) - dy * np.sin(-angle)
        ry = dx * np.sin(-angle) + dy * np.cos(-angle)
        a = side * np.sqrt(3) / 2
        for i in range(6):
            theta = np.pi/6 + i * np.pi / 3  # Edge normals at 30, 90, 150, etc.
            nx, ny = np.cos(theta), np.sin(theta)
            if rx * nx + ry * ny > a:
                return False
        return True
    
    def hexagons_overlap(c1x, c1y, a1, c2x, c2y, a2, side=1.0):
        v1 = hex_vertices(c1x, c1y, side, a1)
        v2 = hex_vertices(c2x, c2y, side, a2)
        for vx, vy in v1:
            if point_in_hexagon(vx, vy, c2x, c2y, side, a2):
                return True
        for vx, vy in v2:
            if point_in_hexagon(vx, vy, c1x, c1y, side, a1):
                return True
        return False
    
    def compute_min_outer_side(inner_positions, outer_cx=0, outer_cy=0, outer_angle=0):
        max_extent = 0
        oa = np.radians(outer_angle)
        for pos in inner_positions:
            verts = hex_vertices(pos[0], pos[1], 1.0, pos[2])
            for vx, vy in verts:
                dx, dy = vx - outer_cx, vy - outer_cy
                rx = dx * np.cos(-oa) - dy * np.sin(-oa)
                ry = dx * np.sin(-oa) + dy * np.cos(-oa)
                for k in range(6):
                    theta = np.pi/6 + k * np.pi / 3
                    nx, ny = np.cos(theta), np.sin(theta)
                    dist = abs(rx * nx + ry * ny)
                    required_S = dist / (np.sqrt(3) / 2)
                    if required_S > max_extent:
                        max_extent = required_S
        return max_extent
    
    def check_valid(inner_positions, outer_cx, outer_cy, outer_side, outer_angle):
        n = len(inner_positions)
        for i in range(n):
            for j in range(i+1, n):
                if hexagons_overlap(inner_positions[i][0], inner_positions[i][1], 
                                   inner_positions[i][2],
                                   inner_positions[j][0], inner_positions[j][1],
                                   inner_positions[j][2]):
                    return False
        for pos in inner_positions:
            verts = hex_vertices(pos[0], pos[1], 1.0, pos[2])
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, outer_cx, outer_cy, outer_side, outer_angle):
                    return False
        return True
    
    s = np.sqrt(3)
    positions = [(0, 0)]
    for i in range(6):
        angle = i * np.pi / 3
        positions.append((s * np.cos(angle), s * np.sin(angle)))
    for i in [0, 1, 2, 3]:
        angle = i * np.pi / 3
        positions.append((2*s * np.cos(angle), 2*s * np.sin(angle)))
    
    initial_inner = [[p[0], p[1], 0.0] for p in positions]
    best_score = compute_min_outer_side(initial_inner)
    best_inner = initial_inner
    
    def objective(params):
        if time.time() > deadline:
            return 1e6
        inner_pos = [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(N)]
        penalty = 0
        for i in range(N):
            for j in range(i+1, N):
                if hexagons_overlap(inner_pos[i][0], inner_pos[i][1], inner_pos[i][2],
                                   inner_pos[j][0], inner_pos[j][1], inner_pos[j][2]):
                    penalty += 1.0
        if penalty > 0:
            return 1e6 + penalty
        return compute_min_outer_side(inner_pos)
    
    x0 = []
    for p in initial_inner:
        x0.extend([p[0], p[1], p[2]])
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 500, 'xatol': 0.01, 'fatol': 0.01})
        if time.time() < deadline and result.fun < best_score:
            best_score = result.fun
            best_inner = [[result.x[3*i], result.x[3*i+1], result.x[3*i+2]] for i in range(N)]
    except Exception:
        pass
    
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        x0_p = list(x0)
        for i in range(len(x0_p)):
            if i % 3 != 2:
                x0_p[i] += np.random.uniform(-0.3, 0.3)
        try:
            result = minimize(objective, x0_p, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 0.01, 'fatol': 0.01})
            if time.time() < deadline and result.fun < best_score:
                best_score = result.fun
                best_inner = [[result.x[3*i], result.x[3*i+1], result.x[3*i+2]] for i in range(N)]
        except Exception:
            pass
    
    return best_inner, (0.0, 0.0), best_score, 0.0
# EVOLVE-BLOCK-END
