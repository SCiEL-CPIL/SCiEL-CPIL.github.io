import math
import numpy as np
import time
from scipy.optimize import linprog, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 15.0

def construct():
    # Check if we are already close to the time limit
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate 11 points on a triangular lattice
    # Axial coordinates (q, r)
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    # Use SCALE = 1.0 for optimal packing (touching is allowed in standard packing)
    # If validity fails, we can increase slightly
    SCALE = 1.0
    
    pts = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r / 2.0) * SCALE
        y = 1.5 * r * SCALE
        pts.append((x, y))
        
    pts = np.array(pts)
    
    # Function to compute minimal side length S for a given outer rotation beta (radians)
    # Using LP to find optimal center and minimal D (apothem of outer hexagon)
    def calc_S_for_beta(beta):
        # Check time limit
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        # Compute M_k (max projection) and m_k (min projection) for each normal
        M_vals = []
        m_vals = []
        for nx, ny in normals:
            projections = pts[:, 0] * nx + pts[:, 1] * ny
            M_vals.append(np.max(projections))
            m_vals.append(np.min(projections))
        
        A = APOTHEM
        
        c_obj = np.array([0.0, 0.0, 1.0])
        
        A_ub = []
        b_ub = []
        
        for k in range(6):
            nx, ny = normals[k]
            M_k = M_vals[k]
            m_k = m_vals[k]
            
            # Constraint 1: n . c + D >= M_k + A
            # -n . c - D <= -(M_k + A)
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-(M_k + A))
            
            # Constraint 2: n . c - D <= m_k - A
            A_ub.append([nx, ny, -1.0])
            b_ub.append(m_k - A)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # D must be at least A (apothem of inner hexagon)
        bounds = [(None, None), (None, None), (A, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx = res.x[0]
                cy = res.x[1]
                S = D / A
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    # Minimize S over beta in [0, pi/6]
    best_S = float('inf')
    best_beta = 0.0
    best_center = None
    
    # Coarse grid search first
    for i in range(25):
        beta = i * (math.pi / 6) / 24
        S, center, D = calc_S_for_beta(beta)
        if S < best_S:
            best_S = S
            best_beta = beta
            best_center = center
    
    # Refine with minimize_scalar
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            # First pass with looser tolerance
            res = minimize_scalar(lambda b: calc_S_for_beta(b)[0], 
                                 bounds=(0, math.pi / 6), method='bounded',
                                 options={'maxiter': 100, 'xatol': 1e-8})
            if res.fun < best_S:
                best_S = res.fun
                best_beta = res.x
                _, best_center, _ = calc_S_for_beta(best_beta)
                
            # Second polish pass with tighter tolerance around the best
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN and best_beta > 0 and best_beta < math.pi / 6:
                eps = 1e-4
                res2 = minimize_scalar(lambda b: calc_S_for_beta(b)[0],
                                      bounds=(max(0, best_beta - eps), min(math.pi / 6, best_beta + eps)),
                                      method='bounded',
                                      options={'maxiter': 50, 'xatol': 1e-10})
                if res2.fun < best_S:
                    best_S = res2.fun
                    best_beta = res2.x
                    _, best_center, _ = calc_S_for_beta(best_beta)
    except Exception:
        pass
    
    if best_center is None or best_S == float('inf'):
        # Fallback: Calculate a valid S using the centroid and a simple bound
        best_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        best_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_S = (max_proj + APOTHEM) / APOTHEM

    # Prepare output
    angle_deg = math.degrees(best_beta)
    
    inner = []
    for x, y in pts:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(best_center[0]), float(best_center[1])]
    side = float(best_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
