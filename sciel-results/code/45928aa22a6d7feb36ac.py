"""Pack 12 unit hexagons in smallest regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = time.time() + 130.0

def _hex_vertices(cx, cy, s, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.linspace(0, 2*np.pi, 7)[:-1]
    return np.column_stack([cx + s*np.cos(angles), cy + s*np.sin(angles)])

def _min_outer_side(positions, angles_deg, outer_center, outer_angle_deg):
    """Minimum outer hexagon side to contain all inner hexagons."""
    oa = np.radians(outer_angle_deg)
    normals = np.column_stack([np.cos(oa + np.arange(6)*np.pi/3), 
                                 np.sin(oa + np.arange(6)*np.pi/3)])
    apothem_factor = np.sqrt(3)/2
    oc = np.array(outer_center)
    max_req = 0.0
    for pos, ang in zip(positions, angles_deg):
        verts = _hex_vertices(pos[0], pos[1], 1.0, ang)
        # For each vertex, for each normal, compute required S
        # dot(v - oc, n_i) <= S * apothem_factor
        # S >= dot(v-oc, n_i) / apothem_factor (if positive)
        for v in verts:
            dots = np.dot(v - oc, normals.T)  # shape (6,)
            reqs = np.maximum(dots, 0.0) / apothem_factor
            max_req = max(max_req, np.max(reqs))
    return max_req

def _hex_grid(cx, cy, spacing, rows, cols):
    positions = []
    for i in range(rows):
        for j in range(cols):
            x = cx + j * spacing * 1.5
            y = cy + i * spacing * np.sqrt(3) + (j % 2) * spacing * np.sqrt(3)/2
            positions.append([x, y, 0.0])
    return positions[:N]

def _ring(cx, cy, r):
    positions = [[cx, cy, 0.0]]
    for k in range(6):
        a = k * np.pi/3
        positions.append([cx + r*np.cos(a), cy + r*np.sin(a), 0.0])
    for k in range(5):
        a = k * 2*np.pi/5 + np.pi/5
        positions.append([cx + 2*r*np.cos(a), cy + 2*r*np.sin(a), 0.0])
    return positions[:N]

# EVOLVE-BLOCK-START
def construct():
    global DEADLINE
    DEADLINE = time.time() + 130.0
    best_side = float('inf')
    best_inner = None
    
    inits = [
        _hex_grid(0, 0, 2.0, 2, 6),
        _hex_grid(0, 0, 2.0, 3, 4),
        _hex_grid(0, 0, 2.0, 4, 3),
        _ring(0, 0, 2.0),
        _ring(0, 0, 2.2),
        [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N-N//2)],
    ]
    
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    for init in inits:
        if time.time() > DEADLINE:
            break
        
        x0 = []
        for p in init:
            x0.extend([p[0], p[1], np.radians(p[2])])
        
        def obj(x):
            if time.time() > DEADLINE:
                return best_side
            pos = np.array(x[0::3])
            ys = np.array(x[1::3])
            angs_rad = np.array(x[2::3])
            angs_deg = np.degrees(angs_rad)
            return _min_outer_side(np.column_stack([pos, ys]), angs_deg, outer_center, outer_angle)
        
        result = minimize(obj, x0, method='Nelder-Mead', 
                         options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True})
        
        if result.fun < best_side:
            best_side = result.fun
            x = result.x
            best_inner = [[x[i*3], x[i*3+1], np.degrees(x[i*3+2])] for i in range(N)]
    
    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N-N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, outer_center, best_side, outer_angle
# EVOLVE-BLOCK-END
