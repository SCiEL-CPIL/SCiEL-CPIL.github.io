"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    
    def hex_verts(cx, cy, side, angle_deg):
        angle = np.radians(angle_deg)
        return np.array([[cx + side*np.cos(angle + i*np.pi/3), 
                         cy + side*np.sin(angle + i*np.pi/3)] for i in range(6)])
    
    def max_violation(verts, cx, cy, side, angle_deg):
        """Return max amount by which vertices violate the outer hexagon boundary."""
        angle = np.radians(angle_deg)
        apothem = side * np.sqrt(3) / 2
        max_v = 0
        for v in verts:
            for j in range(6):
                aj = angle + j * np.pi / 3 + np.pi / 6
                proj = (v[0] - cx) * np.cos(aj) + (v[1] - cy) * np.sin(aj)
                if proj > apothem:
                    max_v = max(max_v, proj - apothem)
        return max_v
    
    def objective(params):
        side, cx, cy, angle = params
        total_violation = 0
        for i in range(len(inner_hexes)):
            ix, iy, ia = inner_hexes[i]
            verts = hex_verts(ix, iy, 1.0, ia)
            total_violation += max_violation(verts, cx, cy, side, angle) ** 2
        return side + 100.0 * total_violation
    
    # Generate hexagonal arrangement
    positions = [[0.0, 0.0, 0.0]]
    spacing = np.sqrt(3)
    
    for ring in range(1, 10):
        for i in range(6 * ring):
            if len(positions) >= N:
                break
            side_idx = i // ring
            pos_on_side = i % ring
            base_angle = side_idx * np.pi / 3
            
            x = spacing * ring * np.cos(base_angle)
            y = spacing * ring * np.sin(base_angle)
            
            side_angle = base_angle + np.pi / 2
            offset = (pos_on_side - (ring - 1) / 2.0) * spacing
            
            x += offset * np.cos(side_angle)
            y += offset * np.sin(side_angle)
            
            positions.append([x, y, 0.0])
        if len(positions) >= N:
            break
    
    inner_hexes = positions[:N]
    
    # Estimate initial outer hexagon
    xs = [p[0] for p in inner_hexes]
    ys = [p[1] for p in inner_hexes]
    cx0 = (min(xs) + max(xs)) / 2
    cy0 = (min(ys) + max(ys)) / 2
    max_dist = max(np.sqrt((p[0]-cx0)**2 + (p[1]-cy0)**2) for p in inner_hexes)
    side0 = max_dist + 1.5
    
    x0 = [side0, cx0, cy0, 0.0]
    best_side = side0
    best_params = x0.copy()
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-7})
        if res.fun < best_side and res.x[0] > 0:
            # Verify no violations
            side, cx, cy, angle = res.x
            viol = 0
            for i in range(len(inner_hexes)):
                ix, iy, ia = inner_hexes[i]
                verts = hex_verts(ix, iy, 1.0, ia)
                viol += max_violation(verts, cx, cy, side, angle)
            if viol < 1e-6:
                best_side = side
                best_params = res.x.copy()
    except Exception:
        pass
    
    inner = [[p[0], p[1], p[2]] for p in inner_hexes]
    return inner, (best_params[1], best_params[2]), best_params[0], best_params[3]
# EVOLVE-BLOCK-END
