"""Packing 12 unit hexagons in a minimal regular hexagon."""
import time
import numpy as np
from scipy.optimize import minimize

N = 12
T0 = time.time()
DEADLINE = 150.0

def hex_vertices(cx, cy, s, angle_deg):
    a = np.radians(angle_deg)
    angles = np.arange(6) * np.pi / 3.0 + a
    return cx + s * np.cos(angles), cy + s * np.sin(angles)

def outer_side_for_packing(inner_pts, inner_angles, outer_angle=0.0):
    """Compute minimum outer hexagon side S (centered at origin, rotated outer_angle)
    such that all inner unit hexagons fit inside."""
    a_out = np.radians(outer_angle)
    # The outer hexagon in local frame (before rotation) has half-plane normals at 30,90,...
    # After rotation by a_out, normals are at 30+a_out, 90+a_out, ...
    thetas = [np.radians(30 + 60 * k) + a_out for k in range(6)]
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    
    max_val = 0.0
    for i in range(len(inner_pts)):
        cx, cy = inner_pts[i]
        vx, vy = hex_vertices(cx, cy, 1.0, inner_angles[i])
        for j in range(6):
            x, y = vx[j], vy[j]
            for k in range(6):
                d = x * cos_thetas[k] + y * sin_thetas[k]
                if d > max_val:
                    max_val = d
    S_required = max_val / (np.sqrt(3) / 2.0)
    return S_required

def build_initial_positions():
    """Hexagonal lattice arrangement of 12 flat-top unit hexagons."""
    dx = np.sqrt(3)
    dy = 1.5
    positions = []
    angles = []
    for row in range(3):
        for col in range(4):
            x = col * dx
            y = row * dy
            if row % 2 == 1:
                x += dx / 2.0
            positions.append((x, y))
            angles.append(0.0)
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    positions = [(x - cx, y - cy) for x, y in positions]
    return positions, angles

def objective(params):
    inner_pts = []
    inner_angles = []
    for i in range(N):
        inner_pts.append((params[3*i], params[3*i+1]))
        inner_angles.append(params[3*i+2])
    outer_angle = params[36]
    return outer_side_for_packing(inner_pts, inner_angles, outer_angle)

# EVOLVE-BLOCK-START
def construct():
    global T0, DEADLINE
    
    positions, angles = build_initial_positions()
    
    x0 = []
    for i in range(N):
        x0.append(positions[i][0])
        x0.append(positions[i][1])
        x0.append(angles[i])
    x0.append(0.0)
    
    best_S = outer_side_for_packing(positions, angles, 0.0)
    best_params = x0[:]
    
    # Nelder-Mead
    try:
        if time.time() - T0 < DEADLINE - 40:
            res = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
    except:
        pass
    
    # COBYLA
    try:
        if time.time() - T0 < DEADLINE - 25:
            res = minimize(objective, x0, method='COBYLA',
                         options={'maxiter': 8000, 'rhobeg': 0.05})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
    except:
        pass
    
    # L-BFGS-B
    try:
        if time.time() - T0 < DEADLINE - 10:
            res = minimize(objective, x0, method='L-BFGS-B',
                         options={'maxiter': 5000, 'gtol': 1e-8})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
    except:
        pass
    
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    outer_angle = float(best_params[36])
    
    final_S = outer_side_for_packing(
        [(p[0], p[1]) for p in inner],
        [p[2] for p in inner],
        outer_angle
    )
    
    return inner, (0.0, 0.0), float(final_S), outer_angle
# EVOLVE-BLOCK-END
