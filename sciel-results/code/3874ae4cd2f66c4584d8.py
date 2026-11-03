import numpy as np
import time
from scipy.optimize import minimize

N = 11
MIN_DIST = np.sqrt(3)

def hexagon_boundary_violation(p, s):
    norms = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    apothem = s * np.sqrt(3) / 2
    dots = p @ norms.T
    viol = np.max(dots - apothem)
    return max(0.0, viol)

def energy(points, s):
    E = 0.0
    N = len(points)
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(points[i] - points[j])
            if d < MIN_DIST:
                E += (MIN_DIST - d)**2
    for i in range(N):
        v = hexagon_boundary_violation(points[i], s)
        E += v**2
    return E

def can_pack(s, time_limit):
    start_time = time.time()
    best_energy = float('inf')
    best_points = None
    
    # Honeycomb initialization
    base_pts = [
        [0.0, 0.0],
        [np.sqrt(3), 0.0],
        [np.sqrt(3)/2, 1.5],
        [-np.sqrt(3)/2, 1.5],
        [-np.sqrt(3), 0.0],
        [-np.sqrt(3)/2, -1.5],
        [np.sqrt(3)/2, -1.5]
    ]
    
    for _ in range(30): # 30 restarts
        if time.time() - start_time > time_limit * 0.8:
            break
            
        scale = min(1.0, s / 2.0)
        pts = [[x*scale, y*scale] for x, y in base_pts]
        
        for _ in range(4):
            x = np.random.uniform(-s, s)
            y = np.random.uniform(-s*np.sqrt(3)/2, s*np.sqrt(3)/2)
            pts.append([x, y])
            
        pts = np.array(pts)
        
        def obj(pts_flat):
            p = pts_flat.reshape(-1, 2)
            return energy(p, s)
        
        try:
            res = minimize(obj, pts.flatten(), method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_energy:
                best_energy = res.fun
                best_points = res.x.reshape(-1, 2)
                if best_energy < 1e-4:
                    break
        except:
            pass

    if best_energy < 1e-4:
        return True, best_points
    return False, None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    lo = 2.0
    hi = 4.0
    best_s = hi
    best_pts = None
    
    # Coarse search
    while (hi - lo) > 0.05 and time.time() < deadline - 20.0:
        mid = (lo + hi) / 2.0
        ok, pts = can_pack(mid, time_limit=10.0)
        if ok:
            hi = mid
            best_s = mid
            best_pts = pts
        else:
            lo = mid
            
    # Fine search
    if time.time() < deadline - 5.0:
        lo = best_s - 0.05
        hi = best_s + 0.05
        while (hi - lo) > 0.005 and time.time() < deadline - 5.0:
            mid = (lo + hi) / 2.0
            ok, pts = can_pack(mid, time_limit=5.0)
            if ok:
                hi = mid
                best_s = mid
                best_pts = pts
            else:
                lo = mid
                
    if best_pts is None:
        # Fallback: large hexagon
        S = 10.0
        inner = []
        for i in range(N):
            inner.append([0.0, 0.0, 0.0])
        return inner, [0.0, 0.0], S, 0.0

    S = best_s + 1.0
    inner = []
    for p in best_pts:
        inner.append([p[0], p[1], 0.0])
        
    return inner, [0.0, 0.0], S, 0.0
