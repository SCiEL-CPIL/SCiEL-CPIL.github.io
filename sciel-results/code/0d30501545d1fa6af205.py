import numpy as np
from scipy.optimize import minimize
import time

N = 11

def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    
    np.random.seed(123)
    
    # Hexagonal lattice initial positions
    lat = np.sqrt(3)
    axial = [(0,0),(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1),(2,0),(2,-1),(1,-2),(-2,0)]
    pts = np.array([[lat*(q+r/2.0), 1.5*r] for q,r in axial])
    
    max_r = np.max(np.hypot(pts[:,0], pts[:,1]))
    S0 = max_r + 2.5
    
    # params: N*(x,y,angle_rad) + (cx, cy, phi_rad, S)
    x0 = np.concatenate([pts[:,0], pts[:,1], np.zeros(N), [0.0, 0.0, 0.0, S0]])
    
    def obj(p):
        if time.time() > deadline:
            return 1e10
        ip = p[:N*3].reshape(N, 3)
        cx, cy, phi, S = p[N*3], p[N*3+1], p[N*3+2], p[N*3+3]
        if S < 0.5:
            return 1e10
        apothem = S * np.sqrt(3) / 2.0
        n_ang = phi + np.arange(6) * np.pi / 3.0 + np.pi / 6.0
        normals = np.column_stack([np.cos(n_ang), np.sin(n_ang)])
        
        max_viol = 0.0
        for i in range(N):
            xi, yi, ai = ip[i]
            v_ang = ai + np.arange(6) * np.pi / 3.0
            verts = np.column_stack([xi + np.cos(v_ang), yi + np.sin(v_ang)])
            d = verts - np.array([cx, cy])
            viol = np.max(np.dot(d, normals.T) - apothem)
            max_viol = max(max_viol, viol)
        
        return S + 10.0 * max(0.0, max_viol)
    
    best = x0.copy()
    best_val = obj(x0)
    
    for it in range(6):
        if time.time() > deadline:
            break
        try:
            res = minimize(obj, x0, method='SLSQP', options={'maxiter': 400, 'ftol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best = res.x.copy()
            if it < 5:
                noise = np.random.normal(0, 0.2, size=best.shape)
                noise[-1] = 0  # don't perturb S
                x0 = best + noise
                x0[-1] = max(x0[-1], 1.0)
        except Exception:
            pass
    
    ip = best[:N*3].reshape(N, 3)
    cx, cy, phi, S = best[N*3], best[N*3+1], best[N*3+2], best[N*3+3]
    inner = [[float(ip[i,0]), float(ip[i,1]), float(np.degrees(ip[i,2]))] for i in range(N)]
    return inner, (float(cx), float(cy)), float(S), float(np.degrees(phi))

construct()
