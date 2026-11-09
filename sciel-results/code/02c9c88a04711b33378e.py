import numpy as np
import time
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Hexagonal lattice spacing for unit hexagons
    spacing = np.sqrt(3)
    
    # Compact arrangement of 12 hexagons in hex lattice
    # Use a 3x4 grid in axial coordinates
    axial_coords = []
    for r in range(4):
        for q in range(3):
            axial_coords.append((q, r))
    
    # Convert to cartesian and center
    centers = []
    for q, r in axial_coords:
        x = spacing * (q + r / 2.0)
        y = spacing * np.sqrt(3) / 2.0 * r
        centers.append([x, y])
    
    cx_mean = np.mean([c[0] for c in centers])
    cy_mean = np.mean([c[1] for c in centers])
    centers = [[c[0] - cx_mean, c[1] - cy_mean] for c in centers]
    
    def hex_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.arange(6) * np.pi / 3
        return cx + side * np.cos(angles), cy + side * np.sin(angles)
    
    def max_violation(px, py, hx, hy, side, angle_deg):
        dx = px - hx
        dy = py - hy
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        inradius = side * np.sqrt(3) / 2
        max_v = 0
        for i in range(6):
            theta = i * np.pi / 3
            dist = lx * np.cos(theta) + ly * np.sin(theta) - inradius
            if dist > max_v:
                max_v = dist
        return max_v
    
    def objective(params):
        outer_side = params[0]
        outer_angle = params[1]
        if outer_side < 0.1:
            return 1e10
        max_viol = 0
        for cx, cy in centers:
            vx, vy = hex_vertices(cx, cy, 1.0, 0.0)
            for px, py in zip(vx, vy):
                v = max_violation(px, py, 0.0, 0.0, outer_side, outer_angle)
                if v > max_viol:
                    max_viol = v
        penalty = max(0, max_viol) * 100
        return outer_side + penalty
    
    # Initial guess
    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    max_extent = max(max(abs(x) for x in xs), max(abs(y) for y in ys)) + 1.0
    outer_side_init = max_extent * 2 / np.sqrt(3) + 0.5
    
    x0 = [outer_side_init, 0.0]
    
    best_obj = np.inf
    best_result = None
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9})
        if result.fun < best_obj:
            best_obj = result.fun
            best_result = result.x
    except:
        pass
    
    # Try a few different outer angles
    for angle in [0, 30, 60]:
        x0a = [outer_side_init, float(angle)]
        try:
            result = minimize(objective, x0a, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9})
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result.x
        except:
            pass
    
    if best_result is not None:
        outer_side_opt = best_result[0]
        outer_angle_opt = best_result[1]
    else:
        outer_side_opt = outer_side_init
        outer_angle_opt = 0.0
    
    inner = []
    for cx, cy in centers:
        inner.append([cx, cy, 0.0])
    
    return inner, (0.0, 0.0), outer_side_opt, outer_angle_opt
# EVOLVE-BLOCK-END
