import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 20.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 4.0, 0.0

    # Generate 11 points on a triangular lattice
    # Center (0,0) + 6 in ring 1 + 4 in ring 2
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    pts = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    pts = np.array(pts)
    
    def compute_S(beta, cx, cy):
        """Compute the minimum S for given beta, cx, cy"""
        max_proj = 0.0
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            for px, py in pts:
                proj = abs((px - cx) * nx + (py - cy) * ny)
                max_proj = max(max_proj, proj)
        D = max_proj
        S = D / APOTHEM + 1.0
        return S
    
    def smooth_obj(params, beta_base):
        """Smooth surrogate using log-sum-exp"""
        dbeta, cx, cy = params
        beta = beta_base + dbeta
        
        projs = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            for px, py in pts:
                projs.append(abs((px - cx) * nx + (py - cy) * ny))
        
        projs = np.array(projs)
        temp = 0.05
        scaled = projs / temp
        max_s = np.max(scaled)
        lse = max_s + np.log(np.sum(np.exp(scaled - max_s)))
        D = temp * lse
        return D / APOTHEM + 1.0
    
    best_S = float('inf')
    best_params = None
    
    beta_starts = [0.0, math.pi/6, math.pi/3, math.pi/12, -math.pi/6, -math.pi/12]
    cx_starts = [-0.3, -0.1, 0.0, 0.1, 0.3]
    cy_starts = [-0.3, -0.1, 0.0, 0.1, 0.3]
    
    for beta0 in beta_starts:
        for cx0 in cx_starts:
            for cy0 in cy_starts:
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                
                try:
                    # Phase 1: Smooth optimization
                    res1 = minimize(smooth_obj, [0.0, cx0, cy0], args=(beta0,),
                                   method='Nelder-Mead',
                                   options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-7})
                    
                    if res1.fun >= best_S:
                        continue
                    
                    # Phase 2: Exact polish from smooth result
                    beta_init = beta0 + res1.x[0]
                    cx_init = res1.x[1]
                    cy_init = res1.x[2]
                    
                    res2 = minimize(lambda p: compute_S(p[0], p[1], p[2]),
                                   [beta_init, cx_init, cy_init],
                                   method='Nelder-Mead',
                                   options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12})
                    
                    if res2.fun < best_S:
                        best_S = res2.fun
                        best_params = res2.x
                except Exception:
                    pass
    
    if best_params is None:
        best_S = compute_S(0.0, 0.0, 0.0)
        best_params = [0.0, 0.0, 0.0]
    
    beta, cx, cy = best_params
    
    # Scale points slightly to ensure strict disjointness
    SCALE = 1.0 + 1e-6
    pts_scaled = []
    for x, y in pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    pts_scaled = np.array(pts_scaled)
    
    # Recompute S for scaled points
    max_proj = 0.0
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        for px, py in pts_scaled:
            proj = abs((px - cx) * nx + (py - cy) * ny)
            max_proj = max(max_proj, proj)
    
    S = max_proj / APOTHEM + 1.0
    
    angle_deg = math.degrees(beta)
    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
    
    center = [float(cx), float(cy)]
    side = float(S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
