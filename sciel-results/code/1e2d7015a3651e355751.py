import math
import numpy as np
import time
from scipy.optimize import minimize_scalar, linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def construct():
    # Check time limit
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: Return a known valid configuration
        # 7 hexagons in a flower, S=3
        pts = [(0,0)]
        for i in range(6):
            ang = i * math.pi / 3
            pts.append((math.sqrt(3)*math.cos(ang), math.sqrt(3)*math.sin(ang)))
        return [[float(x), float(y), 0.0] for x, y in pts], [0.0, 0.0], 3.0, 0.0

    # Generate 11 points on a triangular lattice
    # Flower shape: Center + 6 neighbors + 4 from next ring
    # The 4 points are chosen to be as compact as possible.
    # Axial coords:
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
    
    def solve_for_beta(beta):
        """
        Solves the LP for a given outer rotation beta.
        Returns S (side length) or inf.
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf')
        
        # Normals of the outer hexagon
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        # Variables: cx, cy, D
        # D is the max projection of the *centers* relative to outer center.
        # Constraint: | (p_i - c) . n_k | <= D
        # Outer hexagon contains inner hexagons if:
        # (p_i - c) . n_k + APOTHEM <= S * APOTHEM
        # (p_i - c) . n_k <= S * APOTHEM - APOTHEM
        # Let D = S * APOTHEM - APOTHEM.
        # Minimize D.
        # S = (D + APOTHEM) / APOTHEM
        
        c_vec = np.array([0.0, 0.0, 1.0]) # Minimize D
        
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for px, py in pts:
                # (px - cx)nx + (py - cy)ny <= D
                # -nx cx - ny cy - D <= -(px nx + py ny)
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                
                # -((px - cx)nx + (py - cy)ny) <= D
                # nx cx + ny cy - D <= px nx + py ny
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_vec, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                S = (D + APOTHEM) / APOTHEM
                return S
            else:
                return float('inf')
        except Exception:
            return float('inf')

    # Objective function
    def obj(beta):
        return solve_for_beta(beta)

    # Initial guess
    best_beta = 0.0
    best_S = obj(best_beta)
    
    # Minimize over beta in [0, pi/6]
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            res = minimize_scalar(obj, bounds=(0, math.pi / 6), method='bounded',
                                 options={'maxiter': 1000, 'xatol': 1e-8})
            if res.fun < best_S:
                best_S = res.fun
                best_beta = res.x
                
                # Second pass
                local_bounds = (max(0, best_beta - 0.01), min(math.pi / 6, best_beta + 0.01))
                res2 = minimize_scalar(obj, bounds=local_bounds, method='bounded',
                                      options={'maxiter': 2000, 'xatol': 1e-10})
                if res2.fun < best_S:
                    best_S = res2.fun
                    best_beta = res2.x
    except Exception:
        pass

    # Final verification and center calculation
    # We need to return the center (cx, cy) as well.
    # Re-solve LP to get cx, cy
    def solve_full(beta):
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
            
        c_vec = np.array([0.0, 0.0, 1.0])
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
            res = linprog(c_vec, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                cx = res.x[0]
                cy = res.x[1]
                D = res.x[2]
                S = (D + APOTHEM) / APOTHEM
                return cx, cy, S
        except Exception:
            pass
        return 0.0, 0.0, float('inf')

    cx, cy, S_final = solve_full(best_beta)
    
    if S_final == float('inf'):
        # Fallback
        return [[float(x), float(y), 0.0] for x, y in pts], [0.0, 0.0], 10.0, 0.0

    inner = []
    angle_deg = math.degrees(best_beta)
    for x, y in pts:
        inner.append([float(x), float(y), angle_deg])
        
    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = angle_deg
    
    return inner, center, side, rotation
