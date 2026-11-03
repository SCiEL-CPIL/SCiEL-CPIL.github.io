import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate candidate axial coordinate sets
    def generate_candidates():
        candidates = []
        
        # Standard hexagonal lattice patterns
        patterns = [
            # Compact 11-hexagon clusters
            [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (-2,0), (0,2), (0,-2)],
            [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2), (-1,1), (3,1)],
            [(0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1), (0,2), (1,2), (2,2)],
            [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2), (0,3), (1,3)],
            [(0,0), (1,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2), (0,3), (1,3), (2,3)],
            [(0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1), (4,1), (5,1)],
            [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,1), (-2,2), (-1,3), (1,-2)],
            [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (1,2), (2,2), (3,2), (0,3), (1,3)],
            [(0,0), (1,0), (2,0), (1,1), (2,1), (0,2), (1,2), (2,2), (-1,1), (3,1), (0,-1)],
            [(0,0), (1,0), (0,1), (1,1), (2,1), (1,2), (2,2), (0,3), (1,3), (2,3), (3,2)],
            [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (3,1), (0,2), (1,2), (2,2), (1,3)],
            [(0,0), (1,0), (2,0), (3,0), (1,1), (2,1), (3,1), (0,2), (1,2), (2,2), (3,2), (1,3), (2,3)],
            [(0,0), (1,0), (0,1), (1,1), (2,1), (3,1), (0,2), (1,2), (2,2), (1,3), (2,3)],
            [(0,0), (1,0), (2,0), (1,1), (2,1), (0,2), (1,2), (2,2), (0,3), (1,3)],
            [(0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (1,2), (2,2), (3,2)],
            [(0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1), (0,2), (1,2), (2,2)],
            [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2), (0,3), (1,3), (2,3)],
            [(0,0), (1,0), (2,0), (1,1), (2,1), (0,2), (1,2), (2,2), (1,3), (2,3)],
            [(0,0), (1,0), (2,0), (1,1), (2,1), (0,2), (1,2), (2,2), (0,3), (1,3), (2,3)],
            [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (1,2), (2,2), (3,2), (0,3), (1,3)],
            [(0,0), (1,0), (2,0), (3,0), (1,1), (2,1), (3,1), (0,2), (1,2), (2,2), (3,2)],
        ]
        
        for p in patterns:
            if len(p) == 11:
                candidates.append(p)
        
        return candidates

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def calc_S_for_beta(pts, beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        
        n_pts = len(pts)
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for i in range(n_pts):
                px, py = pts[i]
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx = res.x[0]
                cy = res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def exact_S(pts, beta, cx, cy):
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        pts_centered = pts - np.array([cx, cy])
        projs = np.abs(pts_centered @ normals.T)
        max_proj = float(np.max(projs))
        S = (max_proj + APOTHEM) / APOTHEM
        return S

    def newton_center_polish(pts, beta, cx, cy, max_iter=50):
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        
        for _ in range(max_iter):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            pts_c = pts - np.array([cx, cy])
            projs = pts_c @ normals.T
            D = float(np.max(np.abs(projs)))
            if D < 1e-15:
                break
            
            tol = D * 1e-8 + 1e-14
            active_rows = []
            for i in range(len(pts)):
                for k in range(6):
                    if abs(abs(projs[i, k]) - D) < tol:
                        if projs[i, k] > 0:
                            active_rows.append((normals[k][0], normals[k][1], -D + pts[i,0]*normals[k][0] + pts[i,1]*normals[k][1]))
                        else:
                            active_rows.append((-normals[k][0], -normals[k][1], D - pts[i,0]*normals[k][0] - pts[i,1]*normals[k][1]))
            
            if len(active_rows) < 2:
                break
            
            A = np.array([[r[0], r[1]] for r in active_rows])
            b = np.array([r[2] for r in active_rows])
            
            try:
                sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                new_cx, new_cy = float(sol[0]), float(sol[1])
            except Exception:
                break
            
            old_S = (D + APOTHEM) / APOTHEM
            new_pts_c = pts - np.array([new_cx, new_cy])
            new_projs = np.abs(new_pts_c @ normals.T)
            new_D = float(np.max(new_projs))
            new_S = (new_D + APOTHEM) / APOTHEM
            
            if new_S < old_S - 1e-16:
                cx, cy = new_cx, new_cy
                D = new_D
            else:
                tol2 = D * 1e-6 + 1e-12
                active_rows2 = []
                for i in range(len(pts)):
                    for k in range(6):
                        if abs(abs(projs[i, k]) - D) < tol2:
                            if projs[i, k] > 0:
                                active_rows2.append((normals[k][0], normals[k][1], -D + pts[i,0]*normals[k][0] + pts[i,1]*normals[k][1]))
                            else:
                                active_rows2.append((-normals[k][0], -normals[k][1], D - pts[i,0]*normals[k][0] - pts[i,1]*normals[k][1]))
                if len(active_rows2) >= 2:
                    A2 = np.array([[r[0], r[1]] for r in active_rows2])
                    b2 = np.array([r[2] for r in active_rows2])
                    try:
                        sol2, _, _, _ = np.linalg.lstsq(A2, b2, rcond=None)
                        ncx, ncy = float(sol2[0]), float(sol2[1])
                        npc = pts - np.array([ncx, ncy])
                        npd = float(np.max(np.abs(npc @ normals.T)))
                        ns = (npd + APOTHEM) / APOTHEM
                        if ns < old_S - 1e-16:
                            cx, cy = ncx, ncy
                    except Exception:
                        pass
                break
        
        pts_c = pts - np.array([cx, cy])
        D = float(np.max(np.abs(pts_c @ normals.T)))
        return (D + APOTHEM) / APOTHEM, cx, cy

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None
        
        # Coarse grid search over beta
        n_grid = 30
        for i in range(n_grid + 1):
            beta = i * (math.pi / 6) / n_grid
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Refine beta with bounded scalar optimization
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                refine_starts = [best_beta, 0.0, math.pi/12, math.pi/6 * 0.25, math.pi/6 * 0.75]
                for rb in refine_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    lo = max(0, rb - 0.2)
                    hi = min(math.pi/6, rb + 0.2)
                    res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], bounds=(lo, hi), method='bounded',
                                         options={'maxiter': 300, 'xatol': 1e-13})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_beta = res.x
                        _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass
        
        # Final high-precision polish using exact objective
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    return exact_S(pts, beta, cx, cy)
                
                bcx, bcy = best_center
                bb = best_beta
                polish_starts = [[bcx, bcy, bb]]
                for dr in [0.03, -0.03, 0.01, -0.01]:
                    for ang in range(0, 360, 30):
                        dx = dr * math.cos(math.radians(ang))
                        dy = dr * math.sin(math.radians(ang))
                        polish_starts.append([bcx + dx, bcy + dy, bb])
                for db in [-0.02, -0.008, 0.008, 0.02, -0.003, 0.003]:
                    polish_starts.append([bcx, bcy, bb + db])
                
                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                 options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-15,
                                          'adaptive': True})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (float(res.x[0]), float(res.x[1]))
                        best_beta = ((float(res.x[2]) % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass
            
        # Newton active-set center polish
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                bcx, bcy = best_center
                bb = best_beta
                S_n, cx_n, cy_n = newton_center_polish(pts, bb, bcx, bcy, max_iter=80)
                if S_n < best_S:
                    best_S = S_n
                    best_center = (cx_n, cy_n)
        except Exception:
            pass
            
        return best_S, best_beta, best_center

    candidates = generate_candidates()
    
    def hex_radius(axial_list):
        pts = axial_to_cartesian(axial_list)
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        max_proj = 0.0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
            max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
        return max_proj
    
    sorted_indices = sorted(range(len(candidates)), key=lambda i: hex_radius(candidates[i]))
    
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for idx in sorted_indices:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        axial = candidates[idx]
        pts = axial_to_cartesian(axial)
        
        S, beta, center = optimize_shape(pts)
        
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Apply tiny safety scale
    SCALE = 1.0 + 1e-12
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    
    pts_scaled = np.array(pts_scaled)
    
    # Recompute S
    normals_final = np.array([
        (math.cos(best_global_beta + math.pi / 6.0 + k * math.pi / 3.0),
         math.sin(best_global_beta + math.pi / 6.0 + k * math.pi / 3.0))
        for k in range(6)
    ])
    pts_c_final = pts_scaled - np.array([cx, cy])
    D_final = float(np.max(np.abs(pts_c_final @ normals_final.T)))
    S_final = (D_final + APOTHEM) / APOTHEM
    
    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
