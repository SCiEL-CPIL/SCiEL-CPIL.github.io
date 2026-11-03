import numpy as np
from scipy.optimize import minimize
import time

N = 11

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    angle = np.radians(angle_deg)
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + angle
    xs = cx + side * np.cos(base_angles)
    ys = cy + side * np.sin(base_angles)
    return np.column_stack([xs, ys])

def get_all_vertices(positions, angles):
    """Get all vertices of all hexagons."""
    all_verts = []
    for i in range(len(positions)):
        v = hex_vertices(positions[i][0], positions[i][1], angles[i])
        all_verts.append(v)
    return np.vstack(all_verts)

def min_enclosing_hexagon_side(points):
    """
    Find minimum side length L of a regular hexagon containing all points.
    Returns (L, cx, cy, angle_deg).
    """
    best_L = float('inf')
    best_params = None
    
    for angle_deg in np.linspace(0, 60, 13):
        angle = np.radians(angle_deg)
        sqrt3 = np.sqrt(3)
        normal_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180 + angle
        
        def L_for_center(c):
            max_val = 0
            for k in range(6):
                n = np.array([np.cos(normal_angles[k]), np.sin(normal_angles[k])])
                projs = np.dot(points - c, n)
                max_val = max(max_val, np.max(projs))
            return 2 * max_val / sqrt3
        
        c0 = np.mean(points, axis=0)
        
        try:
            res = minimize(
                lambda c: L_for_center(c),
                c0,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-4}
            )
            L_val = L_for_center(res.x)
            if L_val < best_L:
                best_L = L_val
                best_params = (L_val, res.x[0], res.x[1], angle_deg)
        except:
            pass
    
    if best_params is None:
        cx, cy = np.mean(points[:, 0]), np.mean(points[:, 1])
        L = 20.0
        return (L, cx, cy, 0.0)
    
    return best_params

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    sqrt3 = np.sqrt(3)
    
    configs = []
    
    # Config 1: Hexagonal close packing
    pos1 = [(0.0, 0.0)]
    for k in range(6):
        angle = k * np.pi / 3
        pos1.append((sqrt3 * np.cos(angle), sqrt3 * np.sin(angle)))
    for k in range(4):
        angle = k * np.pi / 2 + np.pi / 6
        pos1.append((2 * sqrt3 * np.cos(angle), 2 * sqrt3 * np.sin(angle)))
    configs.append((pos1, [0.0] * N))
    
    # Config 2: Line
    pos2 = [((i - N//2) * sqrt3, 0.0) for i in range(N)]
    configs.append((pos2, [0.0] * N))
    
    # Config 3: Grid
    pos3 = []
    for i in range(3):
        pos3.append(((i - 1) * sqrt3 * 1.1, -1.5 * sqrt3 * 0.9))
    for i in range(4):
        pos3.append(((i - 1.5) * sqrt3 * 0.8, -0.5 * sqrt3 * 0.9))
    for i in range(3):
        pos3.append(((i - 1) * sqrt3 * 1.1, 0.5 * sqrt3 * 0.9))
    pos3.append((0.0, 1.5 * sqrt3 * 0.9))
    configs.append((pos3, [0.0] * N))
    
    best_result = None
    best_L = float('inf')
    
    for pos, angles in configs:
        if time.time() > deadline - 30:
            break
        
        positions = np.array(pos)
        all_verts = get_all_vertices(positions, angles)
        L, cx, cy, angle_deg = min_enclosing_hexagon_side(all_verts)
        
        if L < best_L:
            best_L = L
            best_result = (positions, angles, cx, cy, L, angle_deg)
    
    if best_result is None:
        inner = [[float(i * sqrt3), 0.0, 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 20.0, 0.0
    
    positions, angles, cx, cy, L, angle_deg = best_result
    inner = [[float(positions[i][0]), float(positions[i][1]), float(angles[i])] for i in range(N)]
    return inner, (float(cx), float(cy)), float(L), float(angle_deg)
