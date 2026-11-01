import math
import numpy as np
import time
from scipy.optimize import minimize_scalar, linprog, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Define candidate configurations
    # Axial coordinates (q, r)
    # Center
    center_axial = [(0, 0)]
    # Ring 1
    ring1_axial = [
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)
    ]
    
    # Candidate sets for Ring 2 (need 4 points)
    # Option 1: Seed configuration
    ring2_opt1 = [
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    # Option 2: Symmetric (point symmetry)
    ring2_opt2 = [
        (1, 1), (-1, -1), (0, 2), (0, -2)
    ]
    
    # Option 3: Symmetric
    ring2_opt3 = [
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    
    # Option 4: Symmetric
    ring2_opt4 = [
        (0, 2), (0, -2), (2, -1), (-2, 1)
    ]
    
    # Option 5: Seed configuration but with different 4?
    # Let's try (2,0), (0,2), (-2,2), (-2,0) -> Sum (0, 4)
    ring2_opt5 = [
        (2, 0), (0, 2), (-2, 2), (-2, 0)
    ]

    candidates = [
        center_axial + ring1_axial + ring2_opt1,
        center_axial + ring1_axial + ring2_opt2,
        center_axial + ring1_axial + ring2_opt3,
        center_axial + ring1_axial + ring2_opt4,
        center_axial + ring1_axial + ring2_opt5
    ]
    
    best_result = None
    best_S = float('inf')
    
    for axial_pts in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        # Convert axial to Cartesian
        SCALE = 1.0 + 1e-6
        pts = []
        for q, r in axial_pts:
            x = math.sqrt(3) * (q + r / 2.0) * SCALE
            y = 1.5 * r * SCALE
            pts.append((x, y))
        pts = np.array(pts)
        n_pts = len(pts)
        
        def solve_for_beta(beta):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                return None, None, float('inf')
            
            normals = []
            for k in range(6):
                ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                normals.append((math.cos(ang), math.sin(ang)))
            
            c_obj = np.array([0.0, 0.0, 1.0])
            A_ub = []
            b_ub = []
            
            for nx, ny in normals:
                for px, py in pts:
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
                    cx = res.x[0]
                    cy = res.x[1]
                    D = res.x[2]
                    S = (D + APOTHEM) / APOTHEM
                    return cx, cy, S
                else:
                    return None, None, float('inf')
            except Exception:
                return None, None, float('inf')

        def obj(beta):
            _, _, S = solve_for_beta(beta)
            return S

        # Initial guess
        best_beta = 0.0
        best_cx, best_cy, best_S_local = solve_for_beta(best_beta)
        
        if best_S_local == float('inf'):
            continue

        # Minimize over beta
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(obj, bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 300, 'xatol': 1e-8})
                if res.fun < best_S_local:
                    best_beta = res.x
                    best_cx, best_cy, best_S_local = solve_for_beta(best_beta)
                    
                    # Local polish
                    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                        local_bounds = (max(0, best_beta - 0.02), min(math.pi / 6, best_beta + 0.02))
                        res2 = minimize_scalar(obj, bounds=local_bounds, method='bounded',
                                              options={'maxiter': 500, 'xatol': 1e-10})
                        if res2.fun < best_S_local:
                            best_beta = res2.x
                            best_cx, best_cy, best_S_local = solve_for_beta(best_beta)
        except Exception:
            pass

        if best_S_local < best_S:
            best_S = best_S_local
            best_result = {
                'pts': pts,
                'cx': best_cx,
                'cy': best_cy,
                'beta': best_beta,
                'S': best_S_local
            }

    if best_result is None:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    inner = []
    angle_deg = math.degrees(best_result['beta'])
    for x, y in best_result['pts']:
        inner.append([float(x), float(y), angle_deg])
        
    center = [float(best_result['cx']), float(best_result['cy'])]
    side = float(best_result['S'])
    rotation = angle_deg
    
    return inner, center, side, rotation
