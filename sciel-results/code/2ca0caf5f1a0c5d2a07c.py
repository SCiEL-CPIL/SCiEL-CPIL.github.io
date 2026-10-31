import numpy as np
from scipy.optimize import minimize
import time

N = 11
DEADLINE = 110.0

def hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    return np.array([[cx + side * np.cos(a + k * np.pi / 3), 
                      cy + side * np.sin(a + k * np.pi / 3)] for k in range(6)])

def max_violation_inside(p, center, side, angle_deg):
    a = np.radians(angle_deg)
    max_v = 0.0
    for k in range(6):
        theta = a + np.pi / 6 + k * np.pi / 3
        n = np.array([np.cos(theta), np.sin(theta)])
        d = n @ (p - center)
        v = d - side * np.sqrt(3) / 2
        if v > max_v:
            max_v = v
    return max_v

def objective(x):
    n = N
    inner = x[:3*n].reshape(n, 3)
    ocx, ocy, oside, oangle = x[3*n:]
    
    pen = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt((inner[i,0]-inner[j,0])**2 + (inner[i,1]-inner[j,1])**2)
            if d < 2.0:
                pen += (2.0 - d)**2
    
    for i in range(n):
        verts = hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2])
        for v in verts:
            viol = max_violation_inside(v, ocx, ocy, oside, oangle)
            if viol > 0:
                pen += viol**2
    
    return oside + 50.0 * pen

def make_initial(r):
    inner = [[0.0, 0.0, 0.0]]
    for k in range(6):
        t = k * np.pi / 3
        inner.append([2.0*np.cos(t), 2.0*np.sin(t), 0.0])
    for k in range(4):
        t = np.pi/3 + k * np.pi/3
        inner.append([4.0*np.cos(t), 4.0*np.sin(t), 0.0])
    inner = np.array(inner)
    if r > 0:
        inner = inner + np.random.randn(*inner.shape) * 0.3
    max_r = 0
    for i in range(len(inner)):
        verts = hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2])
        for v in verts:
            max_r = max(max_r, np.sqrt(v[0]**2 + v[1]**2))
    oside = max_r + 0.3
    return np.concatenate([inner.flatten(), [0.0, 0.0, oside, 0.0]])

def construct():
    t0 = time.time()
    best_x, best_score = None, np.inf
    for r in range(5):
        if time.time() - t0 > DEADLINE - 20:
            break
        x0 = make_initial(r)
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': 1500, 'xatol': 1e-5, 'fatol': 1e-5})
        if res.fun < best_score:
            best_score, best_x = res.fun, res.x
    if best_x is None:
        best_x = make_initial(0)
    n = N
    inner = best_x[:3*n].reshape(n, 3)
    ocx, ocy, oside, oangle = best_x[3*n:]
    return ([[float(inner[i,0]), float(inner[i,1]), float(inner[i,2])] for i in range(n)],
            (float(ocx), float(ocy)), float(oside), float(oangle))
