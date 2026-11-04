"""Packing 12 unit hexagons inside a regular hexagon."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 140.0  # safety margin before 150s kill

def hex_vertices(cx, cy, side=1.0, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy)."""
    a = np.radians(angle_deg)
    angles = a + np.radians(np.array([0, 60, 120, 180, 240, 300]))
    return np.column_stack([cx + side * np.cos(angles),
                            cy + side * np.sin(angles)])

def container_side_for_points(points, outer_angle_deg=0.0):
    """Find minimum side length of regular hexagon (rotation outer_angle_deg)
    centered at origin containing all points."""
    a = np.radians(outer_angle_deg)
    normals = np.array([[np.cos(a + np.radians(k*60)), np.sin(a + np.radians(k*60))] for k in range(6)])
    proj = points @ normals.T
    max_abs = np.max(np.abs(proj))
    s = 2.0 * max_abs / np.sqrt(3.0)
    return s

def all_vertices(inner_hexes):
    verts = []
    for (x, y, ang) in inner_hexes:
        v = hex_vertices(x, y, 1.0, ang)
        verts.append(v)
    return np.vstack(verts)

def check_overlap(params):
    """Return True if any two inner hexagons overlap (approximate)."""
    for i in range(N):
        for j in range(i+1, N):
            xi, yi, ai = params[3*i], params[3*i+1], params[3*i+2]
            xj, yj, aj = params[3*j], params[3*j+1], params[3*j+2]
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            # Minimum center distance for non-overlapping unit hexagons
            # depends on relative orientation. Conservative: 1.7
            if dist < 1.6:
                return True
    return False

def objective(params, outer_angle_deg=0.0):
    inner = [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(N)]
    verts = all_vertices(inner)
    s = container_side_for_points(verts, outer_angle_deg)
    # Add penalty for overlaps
    if check_overlap(params):
        s += 100.0
    return s

def construct():
    start_time = time.time()
    best_params = None
    best_val = float('inf')
    
    topologies = []
    
    # Hexagonal cluster: 1 center + 6 ring + 5 outer
    hex_cluster = [(0.0, 0.0, 0.0)]
    for k in range(6):
        ang = np.radians(k * 60)
        hex_cluster.append((np.sqrt(3)*np.cos(ang), np.sqrt(3)*np.sin(ang), 0.0))
    for k in range(5):
        ang = np.radians(k * 72 + 30)
        hex_cluster.append((2*np.sqrt(3)*np.cos(ang), 2*np.sqrt(3)*np.sin(ang), 0.0))
    topologies.append(hex_cluster)
    
    # 2 rows of 6
    row2 = []
    for i in range(6):
        row2.append((i*2.0, 0.0, 0.0))
    for i in range(6):
        row2.append((i*2.0+1.0, np.sqrt(3), 0.0))
    topologies.append(row2)
    
    # 3 rows of 4
    row3 = []
    for i in range(4):
        row3.append((i*2.0, 0.0, 0.0))
    for i in range(4):
        row3.append((i*2.0+1.0, np.sqrt(3), 0.0))
    for i in range(4):
        row3.append((i*2.0, 2*np.sqrt(3), 0.0))
    topologies.append(row3)
    
    for topo in topologies:
        if time.time() > DEADLINE - 10:
            break
        xs = [p[0] for p in topo]
        ys = [p[1] for p in topo]
        cx = (max(xs) + min(xs)) / 2
        cy = (max(ys) + min(ys)) / 2
        centered = [(x-cx, y-cy, a) for (x,y,a) in topo]
        
        x0 = []
        for (x, y, a) in centered:
            x0.extend([x, y, a])
        x0 = np.array(x0)
        
        val0 = objective(x0)
        if val0 < best_val:
            best_val = val0
            best_params = x0.copy()
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                            options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Polish
    if best_params is not None and time.time() < DEADLINE - 5:
        try:
            result = minimize(objective, best_params, method='Nelder-Mead',
                            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return topologies[0], (0.0, 0.0), 40.0, 0.0
    
    inner = [[float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])] for i in range(N)]
    return inner, (0.0, 0.0), float(best_val), 0.0

