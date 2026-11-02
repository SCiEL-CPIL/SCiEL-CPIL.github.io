"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11
DEADLINE = 140.0

def hex_vertices(cx, cy, side, angle_deg):
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3.0
        verts.append([cx + side * np.cos(theta), cy + side * np.sin(theta)])
    return np.array(verts)

def point_in_hex(px, py, hex_cx, hex_cy, hex_side, hex_angle_deg):
    dx = px - hex_cx
    dy = py - hex_cy
    angle = np.radians(hex_angle_deg)
    rx = dx * np.cos(angle) + dy * np.sin(angle)
    ry = -dx * np.sin(angle) + dy * np.cos(angle)
    inradius = hex_side * np.sqrt(3) / 2.0
    max_val = 0.0
    for k in range(6):
        theta = np.radians(30 + k * 60)
        val = rx * np.cos(theta) + ry * np.sin(theta)
        if val > max_val:
            max_val = val
    return max_val <= inradius + 1e-9

def construct():
    start_time = time.time()
    
    hex_dist = np.sqrt(3)
    positions = []
    positions.append([0.0, 0.0])
    for i in range(6):
        theta = i * np.pi / 3.0
        positions.append([hex_dist * np.cos(theta), hex_dist * np.sin(theta)])
    for i in range(4):
        theta = (i + 0.5) * np.pi / 3.0
        positions.append([2 * hex_dist * np.cos(theta), 2 * hex_dist * np.sin(theta)])
    positions = np.array(positions)
    angles = [0.0] * N
    
    def objective(params):
        if time.time() > DEADLINE:
            return 1e10
        inner_params = params[:3*N].reshape(N, 3)
        outer_cx, outer_cy, outer_side, outer_angle = params[3*N:3*N+4]
        if outer_side < 1.0:
            return 1e10
        for i in range(N):
            cx, cy, ang = inner_params[i]
            verts = hex_vertices(cx, cy, 1.0, ang)
            for v in verts:
                if not point_in_hex(v[0], v[1], outer_cx, outer_cy, outer_side, outer_angle):
                    return 1e10
        for i in range(N):
            c1 = np.array([inner_params[i][0], inner_params[i][1]])
            for j in range(i+1, N):
                c2 = np.array([inner_params[j][0], inner_params[j][1]])
                d = np.linalg.norm(c1 - c2)
                if d < np.sqrt(3) - 0.05:
                    return 1e10
                elif d < np.sqrt(3) + 0.05:
                    v1 = hex_vertices(inner_params[i][0], inner_params[i][1], 1.0, inner_params[i][2])
                    v2 = hex_vertices(inner_params[j][0], inner_params[j][1], 1.0, inner_params[j][2])
                    c2x, c2y = inner_params[j][0], inner_params[j][1]
                    for v in v1:
                        if point_in_hex(v[0], v[1], c2x, c2y, 1.0, inner_params[j][2]):
                            return 1e10
                    c1x, c1y = inner_params[i][0], inner_params[i][1]
                    for v in v2:
                        if point_in_hex(v[0], v[1], c1x, c1y, 1.0, inner_params[i][2]):
                            return 1e10
        return outer_side
    
    init_params = []
    for i in range(N):
        init_params.extend([positions[i][0], positions[i][1], angles[i]])
    init_params.extend([0.0, 0.0, 8.0, 0.0])
    init_params = np.array(init_params)
    
    best_side = 1e10
    best_params = init_params.copy()
    
    n_restarts = 3
    for restart in range(n_restarts):
        if time.time() > DEADLINE - 5:
            break
        if restart == 0:
            x0 = init_params
        else:
            x0 = init_params + np.random.randn(len(init_params)) * 0.3
            x0[3*N+2] = max(8.0, x0[3*N+2])
        try:
            result = minimize(objective, x0, method='SLSQP', options={'maxiter': 40, 'ftol': 1e-5})
            if result.fun < 1e9 and result.fun < best_side:
                best_side = result.fun
                best_params = result.x
        except Exception:
            pass
    
    if best_side >= 1e9:
        inner = [[positions[i][0], positions[i][1], angles[i]] for i in range(N)]
        return inner, (0.0, 0.0), 10.0, 0.0
    
    inner_params = best_params[:3*N].reshape(N, 3)
    inner = [[inner_params[i][0], inner_params[i][1], inner_params[i][2]] for i in range(N)]
    outer_cx, outer_cy, outer_side, outer_angle = best_params[3*N:3*N+4]
    return inner, (outer_cx, outer_cy), outer_side, outer_angle

# EVOLVE-BLOCK-START
# construct() is defined above
# EVOLVE-BLOCK-END
