import math
import numpy as np
import time
from scipy.optimize import minimize_scalar, linprog, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def _smooth_min_max(vals, beta=50.0):
    """
    Smooth approximation of max(vals) using log-sum-exp.
    Returns a smooth surrogate for the maximum.
    """
    max_val = np.max(vals)
    if max_val < -100:
        return max_val
    exp_vals = np.exp(beta * (vals - max_val))
    return max_val + math.log(np.sum(exp_vals)) / beta

def _smooth_min(vals, beta=50.0):
    """
    Smooth approximation of min(vals) using negative log-sum-exp.
    Returns a smooth surrogate for the minimum.
    """
    return -_smooth_max(-vals, beta)

def _smooth_max(vals, beta=50.0):
    """
    Smooth approximation of max(vals) using log-sum-exp.
    """
    max_val = np.max(vals)
    if max_val < -100:
        return max_val
    exp_vals = np.exp(beta * (vals - max_val))
    return max_val + math.log(np.sum(exp_vals)) / beta

def construct():
    # Check if we are already close to the time limit
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate multiple candidate configurations
    configurations = []
    
    # Configuration 1: Compact flower shape
    axial1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    configurations.append(axial1)
    
    # Configuration 2: More spread out
    axial2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (0, 2), (-2, 2), (2, -2)
    ]
    configurations.append(axial2)
    
    # Configuration 3: Linear arrangement
    axial3 = [
        (0, 0),
        (1, 0), (2, 0), (3, 0), (4, 0),
        (-1, 0), (-2, 0), (-3, 0), (-4, 0),
        (0, 1), (0, -1)
    ]
    configurations.append(axial3)
    
    best_result = None
    best_S = float('inf')
    
    for config_idx, axial in enumerate(configurations):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        # Use a very small expansion to ensure strict disjointness
        SCALE = 1.0 + 1e-6
        
        pts = []
        for q, r in axial:
            x = math.sqrt(3) * (q + r / 2.0) * SCALE
            y = 1.5 * r * SCALE
            pts.append((x, y))
            
        pts = np.array(pts)
        n_pts = len(pts)
        
        def solve_for_beta(beta, cx_init=0.0, cy_init=0.0):
            """
            Solves the full LP for a given outer rotation beta.
            Returns (cx, cy, S) or (None, None, inf)
            """
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                return None, None, float('inf')
            
            normals = []
            for k in range(6):
                ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                normals.append((math.cos(ang), math.sin(ang)))
            
            c = np.array([0.0, 0.0, 1.0])
            
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
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
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

        # Initial guess: beta = 0
        best_beta = 0.0
        best_cx, best_cy, best_S_local = solve_for_beta(best_beta)
        
        if best_S_local == float('inf'):
            continue

        # Phase 1: Minimize over beta in [0, pi/6] using the LP
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(obj, bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 200, 'xatol': 1e-8})
                if res.fun < best_S_local:
                    best_beta = res.x
                    best_cx, best_cy, best_S_local = solve_for_beta(best_beta)
                    
                    # Second pass with tighter tolerance
                    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                        local_bounds = (max(0, best_beta - 0.02), min(math.pi / 6, best_beta + 0.02))
                        res2 = minimize_scalar(obj, bounds=local_bounds, method='bounded',
                                              options={'maxiter': 300, 'xatol': 1e-10})
                        if res2.fun < best_S_local:
                            best_beta = res2.x
                            best_cx, best_cy, best_S_local = solve_for_beta(best_beta)
        except Exception:
            pass

        # Phase 2: Smooth surrogate optimization over (cx, cy, beta)
        # This uses a smooth approximation of the max constraint to allow gradient-based optimization
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def smooth_objective(vars_):
                    cx, cy, beta = vars_
                    beta = np.clip(beta, 0, math.pi / 6)
                    
                    # Compute the required distance D for each normal direction
                    # For each normal, the constraint is: max over points of (px*nx + py*ny) - cx*nx - cy*ny <= D
                    # So D >= max_k [(px*nx + py*ny) - (cx*nx + cy*ny)]
                    # We want to minimize D, which is equivalent to minimizing the max over all constraints
                    
                    constraints = []
                    for k in range(6):
                        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                        nx = math.cos(ang)
                        ny = math.sin(ang)
                        
                        # For this normal, the required D is max over points of (px*nx + py*ny - cx*nx - cy*ny)
                        vals = []
                        for px, py in pts:
                            val = (px - cx) * nx + (py - cy) * ny
                            vals.append(val)
                        vals = np.array(vals)
                        # The constraint requires D >= max(vals)
                        # We use smooth max as a surrogate
                        constraints.append(_smooth_max(vals, beta=100.0))
                    
                    # D must be >= max of all constraint values
                    D = _smooth_max(np.array(constraints), beta=100.0)
                    S = (D + APOTHEM) / APOTHEM
                    return S
                
                # Initial guess from LP solution
                x0 = np.array([best_cx, best_cy, best_beta])
                
                # Try smooth optimization
                res_smooth = minimize(smooth_objective, x0, method='L-BFGS-B',
                                     bounds=[(None, None), (None, None), (0, math.pi / 6)],
                                     options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
                
                if res_smooth.fun < best_S_local:
                    # Verify with exact LP
                    cx_test, cy_test, S_test = solve_for_beta(res_smooth.x[2])
                    if S_test < best_S_local:
                        best_cx, best_cy, best_S_local = cx_test, cy_test, S_test
                        best_beta = res_smooth.x[2]
        except Exception:
            pass

        # Phase 3: Final high-precision LP polish
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                local_bounds = (max(0, best_beta - 0.005), min(math.pi / 6, best_beta + 0.005))
                res3 = minimize_scalar(obj, bounds=local_bounds, method='bounded',
                                      options={'maxiter': 1000, 'xatol': 1e-12})
                if res3.fun < best_S_local:
                    best_beta = res3.x
                    best_cx, best_cy, best_S_local = solve_for_beta(best_beta)
                    
                # Even tighter
                if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                    local_bounds2 = (max(0, best_beta - 0.001), min(math.pi / 6, best_beta + 0.001))
                    res4 = minimize_scalar(obj, bounds=local_bounds2, method='bounded',
                                          options={'maxiter': 2000, 'xatol': 1e-14})
                    if res4.fun < best_S_local:
                        best_beta = res4.x
                        best_cx, best_cy, best_S_local = solve_for_beta(best_beta)
        except Exception:
            pass

        # Update global best
        if best_S_local < best_S:
            best_S = best_S_local
            best_result = (best_cx, best_cy, best_beta, pts.copy())
    
    # If no valid result found, return fallback
    if best_result is None:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    best_cx, best_cy, best_beta, best_pts = best_result
    
    # Prepare output
    inner = []
    angle_deg = math.degrees(best_beta)
    for x, y in best_pts:
        inner.append([float(x), float(y), angle_deg])
        
    center = [float(best_cx), float(best_cy)]
    side = float(best_S)
    rotation = angle_deg
    
    return inner, center, side, rotation
