import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 15.0
GAP_FACTOR = 1.005

def axial_to_cartesian(axial_list):
    pts = []
    for q, r in axial_list:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    return np.array(pts)

def hexagon_vertices(cx, cy, side, angle_deg):
    a = math.radians(angle_deg)
    angles = a + np.pi/6 + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def construct():
    deadline = START_TIME + TIME_LIMIT - SAFETY_MARGIN
    
    if time.time() > deadline:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def calc_D_for_center(pts, cx, cy, beta):
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        return np.max(np.abs(projs))

    def objective_center(params, pts, beta):
        cx, cy = params
        return calc_D_for_center(pts, cx, cy, beta)

    def optimize_shape(pts):
        best_D = float('inf')
        best_beta = 0.0
        best_center = (0.0, 0.0)
        
        centroid = np.mean(pts, axis=0)
        mec_center = centroid
        
        for i in range(25):
            beta = i * (math.pi / 6) / 24
            best_res_for_beta = None
            for cx0, cy0 in [(mec_center[0], mec_center[1]), 
                             (centroid[0], centroid[1]),
                             (centroid[0] + 0.1, centroid[1]),
                             (centroid[0], centroid[1] + 0.1)]:
                res = minimize(objective_center, [cx0, cy0], args=(pts, beta), method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10})
                if best_res_for_beta is None or res.fun < best_res_for_beta.fun:
                    best_res_for_beta = res
            
            if best_res_for_beta.fun < best_D:
                best_D = best_res_for_beta.fun
                best_beta = beta
                best_center = (best_res_for_beta.x[0], best_res_for_beta.x[1])
        
        def obj_beta(beta):
            cx0, cy0 = best_center
            res = minimize(objective_center, [cx0, cy0], args=(pts, beta), method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-12})
            return res.fun
        
        res_beta = minimize_scalar(obj_beta, bounds=(max(0, best_beta - 0.15), min(math.pi/6, best_beta + 0.15)), method='bounded')
        if res_beta.fun < best_D:
            best_D = res_beta.fun
            best_beta = res_beta.x
            cx0, cy0 = best_center
            res_c = minimize(objective_center, [cx0, cy0], args=(pts, best_beta), method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-14})
            best_center = (res_c.x[0], res_c.x[1])
            best_D = res_c.fun
        
        res_final = minimize(objective_center, list(best_center), args=(pts, best_beta), method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-16, 'fatol': 1e-16})
        if res_final.fun < best_D:
            best_D = res_final.fun
            best_center = (res_final.x[0], res_final.x[1])

        S = (best_D + APOTHEM) / APOTHEM
        return S, best_beta, best_center

    candidates = []
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)])
    candidates.append([(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(-1,1),(2,0),(3,0),(-1,0),(0,-1),(1,-1)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(1,1),(-1,0),(0,-1),(1,-1),(2,-1),(2,0),(2,1)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(2,1),(2,0),(3,0)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(-1,1),(2,1),(0,2),(1,2),(2,2),(-1,0)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(-1,1),(0,2),(1,2),(-1,2),(2,1),(2,0),(2,-1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(0,1),(0,2),(1,2),(2,2),(3,2)])

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for axial in candidates:
        if time.time() > deadline:
            break
        pts = axial_to_cartesian(axial)
        centroid = np.mean(pts, axis=0)
        pts_scaled = centroid + GAP_FACTOR * (pts - centroid)
        
        S, beta, center = optimize_shape(pts_scaled)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts_scaled
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        pts = axial_to_cartesian(candidates[0])
        centroid = np.mean(pts, axis=0)
        pts_scaled = centroid + GAP_FACTOR * (pts - centroid)
        cx, cy = np.mean(pts_scaled, axis=0)
        beta = 0.0
        D = calc_D_for_center(pts_scaled, cx, cy, beta)
        best_global_S = (D + APOTHEM) / APOTHEM
        best_global_pts = pts_scaled
        best_global_beta = beta
        best_global_center = (cx, cy)

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(best_global_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
