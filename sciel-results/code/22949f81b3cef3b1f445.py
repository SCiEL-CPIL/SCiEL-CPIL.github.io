import math
import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    # Check if we are already close to the time limit
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Initial lattice points (axial to cartesian)
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    pts0 = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts0.append((x, y))
    pts0 = np.array(pts0)

    # Normals of the outer hexagon (aligned with inner for optimal packing)
    # Angles: 30, 90, 150, 210, 270, 330 degrees
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    normals = np.column_stack([np.cos(angles), np.sin(angles)])

    # Objective function for smooth optimization
    # Variables: p_1...p_N (2N), c_x, c_y, S
    # Total 2N + 3 = 25 variables
    def objective(x):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return 1e10
        
        # Unpack
        p = x[:2*N].reshape(N, 2)
        cx, cy = x[2*N], x[2*N+1]
        S = x[2*N+2]
        
        if S < 3.0: S = 3.0 # Lower bound
        
        # 1. Containment Penalty
        # (p_i - c) . n_k <= S * APOTHEM - APOTHEM
        limit = S * APOTHEM - APOTHEM
        diff = p - np.array([cx, cy])
        projs = diff @ normals.T # (N, 6)
        
        # Positive violations
        viol_pos = np.maximum(0, projs - limit)
        # Negative violations
        viol_neg = np.maximum(0, -projs - limit)
        
        pen_contain = np.sum(viol_pos**2) + np.sum(viol_neg**2)
        
        # 2. Overlap Penalty
        # Two hexagons i, j overlap if dist < 2 * APOTHEM in all normal directions?
        # Minkowski difference is hexagon with apothem 2*APOTHEM = sqrt(3).
        # So they overlap if |d . n_k| < sqrt(3) for ALL k.
        # We want to penalize if this condition is true.
        # Let m_k = |d . n_k|.
        # If min(m_k) < sqrt(3), they overlap.
        # Penalty = max(0, sqrt(3) - min(m_k))^2
        
        pen_overlap = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = p[i] - p[j]
                m = np.abs(d @ normals.T) # (6,)
                min_m = np.min(m)
                if min_m < math.sqrt(3):
                    pen_overlap += (math.sqrt(3) - min_m)**2
        
        # 3. Regularization to keep points close to initial lattice (optional, helps convergence)
        # But we want to move them. Maybe small weight.
        pen_reg = 0.01 * np.sum((p - pts0)**2)
        
        return S + 10.0 * pen_contain + 10.0 * pen_overlap + pen_reg

    def objective_grad(x):
        # Gradient is complex, use numerical or L-BFGS-B with finite diff
        # Given time, use finite diff or BFGS
        return objective(x)

    # Initial guess
    x0 = np.zeros(2*N + 3)
    x0[:2*N] = pts0.flatten()
    x0[2*N] = 0.0
    x0[2*N+1] = 0.0
    x0[2*N+2] = 5.0

    best_S = float('inf')
    best_x = x0

    # Run optimization
    try:
        res = minimize(objective, x0, method='L-BFGS-B', 
                       options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.fun < best_S:
            best_S = res.fun
            best_x = res.x
    except Exception:
        pass

    # Polish with LP
    # Extract best positions
    p_best = best_x[:2*N].reshape(N, 2)
    
    # Solve LP for exact S, cx, cy given p_best
    # Minimize S
    # s.t. (p_i - c) . n_k <= S * APOTHEM - APOTHEM
    #      -(p_i - c) . n_k <= S * APOTHEM - APOTHEM
    # Variables: cx, cy, S
    c_obj = np.array([0, 0, 1])
    A_ub = []
    b_ub = []
    
    for nx, ny in normals:
        for px, py in p_best:
            # (px - cx)nx + (py - cy)ny <= S * APOTHEM - APOTHEM
            # px*nx - cx*nx + py*ny - cy*ny - S*APOTHEM <= -APOTHEM
            # -nx*cx - ny*cy - APOTHEM*S <= -APOTHEM - px*nx - py*ny
            A_ub.append([-nx, -ny, -APOTHEM])
            b_ub.append(-APOTHEM - px*nx - py*ny)
            
            # -(px - cx)nx - (py - cy)ny <= S * APOTHEM - APOTHEM
            # -px*nx + cx*nx - py*ny + cy*ny - S*APOTHEM <= -APOTHEM
            # nx*cx + ny*cy - APOTHEM*S <= -APOTHEM + px*nx + py*ny
            A_ub.append([nx, ny, -APOTHEM])
            b_ub.append(-APOTHEM + px*nx + py*ny)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res_lp = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res_lp.status == 0:
            cx_opt, cy_opt, S_opt = res_lp.x
            if S_opt < best_S:
                best_S = S_opt
                best_cx, best_cy = cx_opt, cy_opt
            else:
                best_cx, best_cy = cx_opt, cy_opt
        else:
            best_cx, best_cy = 0, 0
            best_S = 5.0 # Fallback
    except Exception:
        best_cx, best_cy = 0, 0
        best_S = 5.0

    # Prepare output
    inner = []
    for i in range(N):
        inner.append([float(p_best[i, 0]), float(p_best[i, 1]), 0.0])
    
    center = [float(best_cx), float(best_cy)]
    side = float(best_S)
    rotation = 0.0

    return inner, center, side, rotation
