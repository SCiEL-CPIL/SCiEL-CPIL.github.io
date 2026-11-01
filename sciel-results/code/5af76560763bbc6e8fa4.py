import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def _remaining():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)

def construct():
    if _remaining() <= 0:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate candidate topologies (axial coordinates)
    # These are sets of 11 hexagons
    candidates = []
    
    # Flower (1 center, 6 around, 4 more)
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)])
    # Row based
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    # Compact
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    # Elongated
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)])
    # Variant
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def calc_S_for_beta(pts, beta):
        if _remaining() <= 0:
            return float('inf'), None, None
        
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for px, py in pts:
                # (px - cx) * nx + (py - cy) * ny <= D
                # -nx*cx - ny*cy - D <= -px*nx - py*ny
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                
                # -(px - cx) * nx - (py - cy) * ny <= D
                # nx*cx + ny*cy - D <= px*nx + py*ny
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
                # Correct S calculation
                S = (2.0 * D) / math.sqrt(3)
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def optimize_shape(pts, do_polish=True):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        if _remaining() <= 0:
            return float('inf'), 0.0, None
            
        # Coarse search for beta
        for i in range(37):
            beta = i * (math.pi / 6) / 36
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
                
        if not do_polish:
            return best_S, best_beta, best_center
            
        try:
            # Fine search for beta
            res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0],
                                  bounds=(0, math.pi / 6), method='bounded',
                                  options={'maxiter': 100, 'xatol': 1e-8})
            if res.fun < best_S:
                best_S = res.fun
                best_beta = res.x
                _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass
            
        return best_S, best_beta, best_center

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for axial in candidates:
        if _remaining() <= 0:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts, do_polish=True)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None:
        # Fallback
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (0.0, 0.0)
        best_global_beta = 0.0
        best_global_S = 10.0

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Construct inner list
    # Note: The problem allows individual angles, but we assume all are 0 relative to the global frame?
    # The return format is (x, y, angle_degrees).
    # If we used a global rotation beta for the container, and the inner hexagons are aligned with the container,
    # then the inner hexagons are rotated by beta relative to the world?
    # No, the "angle_degrees" in the inner list is the rotation of the inner hexagon.
    # The "rotation" in the return is the rotation of the outer hexagon.
    # If the inner hexagons are aligned with the outer hexagon, their angle should be the same as the outer rotation?
    # Or is it relative?
    # Usually, absolute angle.
    # If the outer hexagon is rotated by beta, and the inner hexagons are aligned with it,
    # then the inner hexagons are also rotated by beta.
    
    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(best_global_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
