import numpy as np
from scipy.optimize import minimize
import time

N = 11

def _hex_vertices(cx, cy, s, angle_deg):
    angle_rad = np.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = angle_rad + np.radians(60 * k)
        verts.append((cx + s * np.cos(theta), cy + s * np.sin(theta)))
    return np.array(verts)

def _min_enclosing_hexagon(points):
    def objective(x):
        cx, cy, alpha, s = x
        if s < 0.1:
            s = 0.1
        normals = []
        for k in range(6):
            theta = np.radians(alpha + 30 + 60 * k)
            normals.append([np.cos(theta), np.sin(theta)])
        normals = np.array(normals)
        pts = points - np.array([cx, cy])
        proj = pts @ normals.T
        max_proj = np.max(proj, axis=1)
        inradius = s * np.sqrt(3) / 2
        violations = np.maximum(0, max_proj - inradius)
        return s + 1000 * np.sum(violations ** 2)
    
    cx0 = np.mean(points[:, 0])
    cy0 = np.mean(points[:, 1])
    
    best = None
    best_val = np.inf
    for alpha_init in [0, 30, 60, 90, 120, 150]:
        for s_init in [2, 3, 4, 5]:
            x0 = np.array([cx0, cy0, alpha_init, s_init])
            result = minimize(objective, x0, method='Nelder-Mead',
                            options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-5})
            if result.fun < best_val:
                best_val = result.fun
                best = result.x
    if best is None:
        return 10.0, (cx0, cy0), 0.0
    cx, cy, alpha, s = best
    return s, (cx, cy), alpha

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    v1 = np.array([np.sqrt(3), 0])
    v2 = np.array([np.sqrt(3)/2, 1.5])
    
    lattice = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            pos = i * v1 + j * v2
            lattice.append((pos[0], pos[1]))
    
    lattice_sorted = sorted(lattice, key=lambda p: p[0]**2 + p[1]**2)
    arr = lattice_sorted[:11]
    
    all_verts = []
    for (cx, cy) in arr:
        verts = _hex_vertices(cx, cy, 1.0, 0.0)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    s, center, alpha = _min_enclosing_hexagon(all_verts)
    
    inner = [[cx, cy, 0.0] for (cx, cy) in arr]
    return inner, center, s, alpha
