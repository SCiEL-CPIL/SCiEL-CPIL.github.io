import numpy as np
from scipy.optimize import minimize
import time

N = 11

def _hex_vertices(cx, cy, side, angle_deg):
    angle_rad = np.radians(angle_deg)
    return np.array([[cx + side * np.cos(angle_rad + k * np.pi / 3),
                      cy + side * np.sin(angle_rad + k * np.pi / 3)] for k in range(6)])

def _max_support(points, cx, cy, theta_deg):
    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    theta_rad = np.radians(theta_deg)
    ca, sa = np.cos(-theta_rad), np.sin(-theta_rad)
    x = dx * ca - dy * sa
    y = dx * sa + dy * ca
    max_proj = 0.0
    for k in range(6):
        na = np.radians(30 + k * 60)
        nx, ny = np.cos(na), np.sin(na)
        proj = x * nx + y * ny
        m = np.max(proj)
        if m > max_proj:
            max_proj = m
    return max_proj

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    sqrt3 = np.sqrt(3)
    e1 = np.array([sqrt3, 0.0])
    e2 = np.array([sqrt3 / 2, 1.5])
    
    coords = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),
              (2,0),(1,1),(0,2),(-1,2)]
    positions = np.array([i * e1 + j * e2 for i, j in coords])
    
    all_verts = np.vstack([_hex_vertices(px, py, 1.0, 0.0) for px, py in positions])
    
    def S_needed(cx, cy, theta_deg):
        return _max_support(all_verts, cx, cy, theta_deg) * 2 / sqrt3
    
    def obj(params):
        return S_needed(params[0], params[1], params[2])
    
    best_S = float('inf')
    best_params = np.array([0.0, 0.0, 0.0])
    
    for theta0 in [0, 30, 60, 90, 120]:
        if time.time() > deadline - 30:
            break
        x0 = np.array([0.0, 0.0, float(theta0)])
        try:
            res = minimize(obj, x0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-9})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
        except:
            pass
    
    cx, cy, theta = best_params
    S = S_needed(cx, cy, theta)
    
    inner = [[float(px), float(py), 0.0] for px, py in positions]
    return inner, (float(cx), float(cy)), float(S), float(theta)
