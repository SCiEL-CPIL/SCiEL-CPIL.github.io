import time
import math
import numpy as np
from scipy.optimize import minimize

N = 12

def _edge_normals(theta):
    normals = []
    for k in range(6):
        ang = theta + math.pi / 6.0 + math.pi / 3.0 * k
        normals.append((math.cos(ang), math.sin(ang)))
    return np.array(normals)

def _inner_hexagon_vertices(x, y, ang_deg):
    ang = math.radians(ang_deg)
    verts = []
    for k in range(6):
        a = ang + math.pi / 3.0 * k
        verts.append((x + math.cos(a), y + math.sin(a)))
    return np.array(verts)

def _required_side(inner_verts_all, center, theta):
    normals = _edge_normals(theta)
    apothem = math.sqrt(3.0) / 2.0
    max_proj = -np.inf
    c = np.array(center)
    for verts in inner_verts_all:
        rel = verts - c
        proj = rel @ normals.T
        m = np.max(proj)
        if m > max_proj:
            max_proj = m
    s = max_proj / apothem
    return max(s, 0.0)

def _objective(params):
    n_inner = 3 * N
    inner_params = params[:n_inner]
    cx = params[n_inner]
    cy = params[n_inner + 1]
    theta = params[n_inner + 2]
    
    inner_verts_all = []
    for i in range(N):
        x = inner_params[3*i]
        y = inner_params[3*i+1]
        ang = inner_params[3*i+2]
        inner_verts_all.append(_inner_hexagon_vertices(x, y, ang))
    
    return _required_side(inner_verts_all, (cx, cy), theta)

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    
    a = math.sqrt(3)
    h = a * math.sqrt(3) / 2
    
    initial_inner = []
    for row in range(3):
        y = row * h
        if row % 2 == 0:
            xs = [-1.5*a, -0.5*a, 0.5*a, 1.5*a]
        else:
            xs = [-a, 0.0, a, 2.0*a]
        for x in xs:
            initial_inner.append([x, y, 0.0])
    
    cx0 = np.mean([p[0] for p in initial_inner])
    cy0 = np.mean([p[1] for p in initial_inner])
    for p in initial_inner:
        p[0] -= cx0
        p[1] -= cy0
    
    params0 = []
    for p in initial_inner:
        params0.extend([p[0], p[1], p[2]])
    params0.extend([0.0, 0.0, 0.0])
    
    best_params = np.array(params0, dtype=float)
    best_s = _objective(best_params)
    
    try:
        result = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        if result.fun < best_s:
            best_s = result.fun
            best_params = result.x
    except Exception:
        pass
    
    t_mid = time.time()
    if t_mid < deadline - 30:
        try:
            result2 = minimize(_objective, best_params, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-15, 'adaptive': True})
            if result2.fun < best_s:
                best_s = result2.fun
                best_params = result2.x
        except Exception:
            pass
    
    inner = []
    for i in range(N):
        x = float(best_params[3*i])
        y = float(best_params[3*i+1])
        ang = float(best_params[3*i+2])
        inner.append([x, y, ang])
    
    cx = float(best_params[3*N])
    cy = float(best_params[3*N + 1])
    theta = float(best_params[3*N + 2])
    s = float(best_s)
    theta_deg = math.degrees(theta)
    
    return inner, (cx, cy), s, theta_deg
