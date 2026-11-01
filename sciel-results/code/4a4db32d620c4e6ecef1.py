import math
import numpy as np
import time
from scipy.optimize import minimize_scalar, linprog, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # 11-cell polyhex: center + 6 neighbors + 4 from ring 2 (compact flower)
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    # Small expansion for strict disjointness
    SCALE = 1.0 + 1e-7
    
    pts = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r / 2.0) * SCALE
        y = 1.5 * r * SCALE
        pts.append((x, y))
    pts = np.array(pts)
    
    # Inner hexagon vertices (flat-top, angle 0)
    inner_verts = np.array([[math.cos(math.pi/6 + k*math.pi/3), math.sin(math.pi/6 + k*math.pi/3)] for k in range(6)])
    inner_normals = inner_verts  # For regular hex, normals align with vertices
    
    def inner_support(n):
        """Support function of unit inner hexagon in direction n."""
        return float(np.max(inner_normals @ n))
    
    def outer_support(n, S):
        """Support function of outer hexagon (side S) in direction n."""
        return S * APOTHEM  # Apothem of outer hex in any of its face normal directions
    
    def solve_for_beta(beta):
        """
        Solve LP for given outer rotation beta.
        Outer hex face normals at angles: beta + pi/6 + k*pi/3
        Constraint: (p_i - c)·n_k + h_inner(n_k) <= S * h_outer(n_k)
        where h_outer(n_k) = S * APOTHEM (since n_k is a face normal of outer hex)
        So: (p_i - c)·n_k <= S*APOTHEM - h_inner(n_k)
        
        We minimize S. Variables: cx, cy, S
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return None, None, float('inf')
        
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append(np.array([math.cos(ang), math.sin(ang)]))
        
        # For each normal n_k and each point p_i:
        # (p_i - c)·n_k + h_inner(n_k) <= S * APOTHEM
        # p_i·n_k - c·n_k + h_inner(n_k) <= S * APOTHEM
        # -n_k·c - APOTHEM*S <= -(p_i·n_k + h_inner(n_k))
        # i.e., -nx*cx - ny*cy - APOTHEM*S <= -(px*nx + py*ny + h_inner(n_k))
        
        # Also need the constraint from the other side (the hexagon is bounded on all sides)
        # Actually, the 6 face normals already bound all directions for a hexagon.
        # But we need |projection| bounded, which means both n_k and -n_k.
        # However, for a hexagon, the 6 face normals at 60° apart already cover all directions.
        # The support in direction -n_k is handled by the opposite face normal n_{k+3}.
        
        # So we just need: for each k, for each i:
        # (p_i - c)·n_k + h_inner(n_k) <= S * APOTHEM
        
        c_obj = np.array([0.0, 0.0, 1.0])  # Minimize S
        
        A_ub = []
        b_ub = []
        
        for nk in normals:
            nx, ny = nk
            h_in = inner_support(nk)
            for px, py in pts:
                # -nx*cx - ny*cy - APOTHEM*S <= -(px*nx + py*ny + h_in)
                A_ub.append([-nx, -ny, -APOTHEM])
                b_ub.append(-(px*nx + py*ny + h_in))
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                cx = res.x[0]
                cy = res.x[1]
                S = res.x[2]
                return cx, cy, S
            else:
                return None, None, float('inf')
        except Exception:
            return None, None, float('inf')
    
    def smooth_obj(beta, cx, cy, S):
        """
        Smooth approximation using log-sum-exp of the constraint violations.
        f(beta, c, S) = max over (i,k) of [(p_i - c)·n_k + h_inner(n_k) - S*APOTHEM]
        We want this <= 0, and minimize S.
        
        For optimization, we minimize: S + penalty * soft_max(violations)
        where violations = (p_i - c)·n_k + h_inner(n_k) - S*APOTHEM
        """
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append(np.array([math.cos(ang), math.sin(ang)]))
        
        violations = []
        for nk in normals:
            nx, ny = nk
            h_in = inner_support(nk)
            for px, py in pts:
                v = (px - cx)*nx + (py - cy)*ny + h_in - S*APOTHEM
                violations.append(v)
        
        violations = np.array(violations)
        
        # Soft max with temperature
        temp = 0.01
        max_v = np.max(violations)
        if max_v > 1e-10:
            log_sum_exp = max_v + temp * math.log(np.sum(np.exp((violations - max_v) / temp)))
        else:
            log_sum_exp = 0.0
        
        # Objective: S + penalty for violations
        penalty = 100.0
        return S + penalty * log_sum_exp
    
    # Phase 1: Coarse grid search over beta in [0, pi/6]
    best_beta = 0.0
    best_cx, best_cy, best_S = float('inf'), float('inf'), float('inf')
    
    n_coarse = 24
    for i in range(n_coarse + 1):
        beta = i * (math.pi / 6.0) / n_coarse
        cx, cy, S = solve_for_beta(beta)
        if S < best_S:
            best_S = S
            best_beta = beta
            best_cx, best_cy = cx, cy
    
    if best_S == float('inf'):
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    # Phase 2: Fine minimize_scalar
    def obj(beta):
        _, _, S = solve_for_beta(beta)
        return S
    
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            # Multiple starts around the best
            for center_beta in [best_beta, best_beta - 0.05, best_beta + 0.05, math.pi/12]:
                lb = max(0, center_beta - 0.1)
                ub = min(math.pi/6, center_beta + 0.1)
                if lb >= ub:
                    continue
                res = minimize_scalar(obj, bounds=(lb, ub), method='bounded',
                                     options={'maxiter': 200, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_beta = res.x
                    best_cx, best_cy, best_S = solve_for_beta(best_beta)
    except Exception:
        pass
    
    # Phase 3: Nelder-Mead polish on (beta, cx, cy, S) with smooth objective
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        def nm_obj(vars_):
            beta, cx, cy, S = vars_
            return smooth_obj(beta, cx, cy, S)
        
        x0 = np.array([best_beta, best_cx, best_cy, best_S])
        try:
            nm_res = minimize(nm_obj, x0, method='Nelder-Mead',
                             options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 2000, 'maxfev': 2000})
            if nm_res.fun < best_S:
                beta_nm = nm_res.x[0]
                cx_nm = nm_res.x[1]
                cy_nm = nm_res.x[2]
                S_nm = nm_res.x[3]
                # Verify with exact LP
                cx_v, cy_v, S_v = solve_for_beta(beta_nm)
                if S_v < best_S:
                    best_beta = beta_nm
                    best_cx, best_cy, best_S = cx_v, cy_v, S_v
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        try:
            local_lb = max(0, best_beta - 0.001)
            local_ub = min(math.pi/6, best_beta + 0.001)
            res_final = minimize_scalar(obj, bounds=(local_lb, local_ub), method='bounded',
                                       options={'maxiter': 5000, 'xatol': 1e-14})
            if res_final.fun < best_S:
                best_beta = res_final.x
                best_cx, best_cy, best_S = solve_for_beta(best_beta)
        except Exception:
            pass
    
    # Final verification
    if best_S == float('inf'):
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    # Prepare output: inner hexagons at angle 0 (fixed lattice), outer at best_beta
    inner = []
    for x, y in pts:
        inner.append([float(x), float(y), 0.0])
    
    center = [float(best_cx), float(best_cy)]
    side = float(best_S)
    rotation = float(math.degrees(best_beta))
    
    return inner, center, side, rotation
