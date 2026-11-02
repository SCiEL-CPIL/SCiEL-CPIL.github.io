import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    candidates = []
    
    c1 = [(0,0)]
    c1 += [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    c1 += [(2,0), (0,2), (-2,2), (-2,0)]
    candidates.append(c1)
    
    c2 = [(0,0)]
    c2 += [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    c2 += [(2,0), (0,2), (2,-2), (0,-2)]
    candidates.append(c2)
    
    c5 = [(0,0)]
    c5 += [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    c5 += [(2,0), (1,1), (-1,2), (-2,1)]
    candidates.append(c5)
    
    c6 = [(0,0)]
    c6 += [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    c6 += [(2,0), (2,-1), (1,-2), (0,-2)]
    candidates.append(c6)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def objective(params, pts_centered):
        cx, cy, beta = params
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        
        pts_rel = pts_centered - np.array([cx, cy])
        projs = pts_rel @ normals.T
        max_center_proj = np.max(np.abs(projs))
        D = max_center_proj + 1.0
        S = D / APOTHEM
        return S

    best_S = float('inf')
    best_params = None
    best_pts_centered = None

    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        pts = axial_to_cartesian(axial)
        center = np.mean(pts, axis=0)
        pts_centered = pts - center
        
        best_local_S = float('inf')
        best_local_params = None
        
        for start_beta in [0.0, math.pi/6, math.pi/3, math.pi/2, 2*math.pi/3, 5*math.pi/6]:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            init = [0.0, 0.0, start_beta]
            try:
                res = minimize(lambda x: objective(x, pts_centered), init, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < best_local_S:
                    best_local_S = res.fun
                    best_local_params = res.x
            except Exception:
                pass
        
        if best_local_S < best_S:
            best_S = best_local_S
            best_params = best_local_params
            best_pts_centered = pts_centered

    if best_params is None:
        pts = axial_to_cartesian(candidates[0])
        center = np.mean(pts, axis=0)
        best_pts_centered = pts - center
        best_params = [0.0, 0.0, 0.0]
        best_S = objective(best_params, best_pts_centered)

    cx, cy, beta = best_params
    inner_pts = best_pts_centered - np.array([cx, cy])
    
    inner = []
    for x, y in inner_pts:
        inner.append([float(x), float(y), float(math.degrees(beta))])
        
    center = [0.0, 0.0]
    side = float(best_S)
    rotation = float(math.degrees(beta))
    
    return inner, center, side, rotation
