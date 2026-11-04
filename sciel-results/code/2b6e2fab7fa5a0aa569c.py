import numpy as np
from scipy.optimize import minimize

N = 12
EPS = 0.02  # margin for non-overlap

def inner_vertices(phi):
    # 3x4 grid, centered, rotated by phi, each hexagon rotated by phi
    verts = []
    for i in (-1, 0, 1):
        for j in (-1.5, -0.5, 0.5, 1.5):
            cx, cy = 2.0*i, (np.sqrt(3)+EPS)*j
            # rotate center by phi
            c = np.array([cx, cy])
            c = np.array([np.cos(phi)*c[0]-np.sin(phi)*c[1], np.sin(phi)*c[0]+np.cos(phi)*c[1]])
            for k in range(6):
                a = phi + np.pi/3*k
                verts.append([c[0]+np.cos(a), c[1]+np.sin(a)])
    return np.array(verts)

def s_min(params, verts):
    phi, theta, cx, cy = params
    c = np.array([cx, cy])
    # normals
    maxdot = -np.inf
    for i in range(6):
        a = theta + np.pi/6 + np.pi/3*i
        n = np.array([np.cos(a), np.sin(a)])
        dots = verts @ n - (c @ n)
        maxdot = max(maxdot, dots.max())
    return maxdot / np.cos(np.pi/6)

def construct():
    best = None
    best_s = np.inf
    # multi-start
    for phi0 in np.linspace(0, np.pi/3, 7):
        for theta0 in np.linspace(0, np.pi/3, 7):
            verts = inner_vertices(phi0)
            x0 = [phi0, theta0, 0.0, 0.0]
            res = minimize(s_min, x0, args=(verts,), method='Nelder-Mead', options={'maxiter': 2000, 'xatol':1e-6, 'fatol':1e-6})
            s = res.fun
            if s < best_s:
                best_s = s
                best = res.x
    phi, theta, cx, cy = best
    s = best_s
    # build inner list
    inner = []
    for i in (-1,0,1):
        for j in (-1.5,-0.5,0.5,1.5):
            cxl, cyl = 2.0*i, (np.sqrt(3)+EPS)*j
            c = np.array([cxl, cyl])
            c = np.array([np.cos(phi)*c[0]-np.sin(phi)*c[1], np.sin(phi)*c[0]+np.cos(phi)*c[1]])
            inner.append([c[0], c[1], np.degrees(phi)])
    return inner, (cx, cy), s, np.degrees(theta)
