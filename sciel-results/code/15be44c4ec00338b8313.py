import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 145.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    # The optimal configuration for 11 unit hexagons is the "flower" pattern.
    # We will test this and a few other compact shapes.
    configs = []
    
    # Flower: 1 center, 6 ring, 4 in the "corners" of the ring (specifically, filling 4 of the 6 gaps? No, standard flower is 1+6+4? 
    # Actually, the 11-hex polyhex that is most compact is the one with 1 center, 6 around, and 4 more attached to the outer ring to make it convex-ish.
    # Standard "flower" is 7 hexes. Adding 4 more to make 11.
    # Axial coords for 11-hex flower-like shape:
    # Center (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 additions: (2,-1), (1,1), (-1,2), (-2,1) ... wait, this creates a shape.
    # Let's use the specific config from the current program which is known to work well.
    config_flower = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)]
    configs.append(config_flower)
    
    # Another candidate: more linear/strip
    config_strip = [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(0,-1),(1,-1),(2,-1)]
    configs.append(config_strip)
    
    # Another candidate: 2x2 block + extras
    config_block = [(0,0),(1,0),(0,1),(1,1),(2,0),(2,1),(3,0),(3,1),(0,2),(1,2),(2,2)]
    configs.append(config_block)

    best_S_global = float('inf')
    best_result_global = None

    def calc_S_for_beta(pts, beta):
        """LP for minimum enclosing hexagon with given rotation.
        Returns S, center, D."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None

        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])

        n_pts = len(pts)
        n_constraints = 12 * n_pts
        A_ub = np.zeros((n_constraints, 3))
        b_ub = np.zeros(n_constraints)

        for k in range(6):
            nx, ny = normals[k]
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            start = 2 * k * n_pts
            A_ub[start:start+n_pts, 0] = -nx
            A_ub[start:start+n_pts, 1] = -ny
            A_ub[start:start+n_pts, 2] = -1.0
            b_ub[start:start+n_pts] = -projs
            A_ub[start+n_pts:start+2*n_pts, 0] = nx
            A_ub[start+n_pts:start+2*n_pts, 1] = ny
            A_ub[start+n_pts:start+2*n_pts, 2] = -1.0
            b_ub[start+n_pts:start+2*n_pts] = projs

        c_obj = np.array([0.0, 0.0, 1.0])
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

    def exact_S(pts, cx, cy, beta):
        """Exact computation of S for given center and rotation."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        D = np.max(np.abs(projs))
        return (D + APOTHEM) / APOTHEM

    def optimize_config(initial_axial):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return None

        pts = axial_to_cartesian(initial_axial)
        n_pts = len(pts)
        
        # 1. Optimize rotation and center for fixed positions
        best_S_rot = float('inf')
        best_beta_rot = 0.0
        best_center_rot = None
        
        # Grid search for rotation
        n_grid = 60
        for i in range(n_grid):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            beta = i * (math.pi / 6) / (n_grid - 1)
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S_rot:
                best_S_rot = S
                best_beta_rot = beta
                best_center_rot = center
        
        if best_center_rot is None:
            return None

        # Refine rotation
        try:
            window = (math.pi / 6) / (n_grid - 1)
            lo = max(0.0, best_beta_rot - 2 * window)
            hi = min(math.pi / 6, best_beta_rot + 2 * window)
            
            def obj_1d(b):
                S, _, _ = calc_S_for_beta(pts, b)
                return S
            
            res = minimize_scalar(obj_1d, bounds=(lo, hi), method='bounded', options={'maxiter': 100})
            if res.fun < best_S_rot:
                best_S_rot = res.fun
                best_beta_rot = res.x
                _, best_center_rot, _ = calc_S_for_beta(pts, best_beta_rot)
        except Exception:
            pass

        # 2. Optimize positions using Nelder-Mead
        # Objective: S + penalty for overlaps
        def objective(params):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                return float('inf')
            pts_opt = params[:2*n_pts].reshape(-1, 2)
            cx = params[2*n_pts]
            cy = params[2*n_pts+1]
            beta = params[2*n_pts+2]
            beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
            
            S = exact_S(pts_opt, cx, cy, beta)
            
            # Penalty for overlaps
            pen = 0.0
            min_dist = math.sqrt(3) # Distance between centers of touching unit hexagons
            for i in range(n_pts):
                for j in range(i+1, n_pts):
                    dx = pts_opt[i,0] - pts_opt[j,0]
                    dy = pts_opt[i,1] - pts_opt[j,1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < min_dist:
                        pen += (min_dist - dist)**2 * 1000.0
            return S + pen

        initial_params = np.concatenate([pts.ravel(), [best_center_rot[0], best_center_rot[1], best_beta_rot]])
        
        try:
            res = minimize(objective, initial_params, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < 1e10:
                res_pts = res.x[:2*n_pts].reshape(-1, 2)
                res_cx = res.x[2*n_pts]
                res_cy = res.x[2*n_pts+1]
                res_beta = ((res.x[2*n_pts+2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                
                # Check validity
                valid = True
                min_dist = math.sqrt(3)
                for i in range(n_pts):
                    for j in range(i+1, n_pts):
                        dx = res_pts[i,0] - res_pts[j,0]
                        dy = res_pts[i,1] - res_pts[j,1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < min_dist - 1e-6:
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    S_exact = exact_S(res_pts, res_cx, res_cy, res_beta)
                    return {
                        'S': S_exact,
                        'beta': res_beta,
                        'center': (res_cx, res_cy),
                        'pts': res_pts
                    }
        except Exception:
            pass
        
        # Fallback to rotation-only result
        return {
            'S': best_S_rot,
            'beta': best_beta_rot,
            'center': best_center_rot,
            'pts': pts
        }

    for cfg in configs:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        result = optimize_config(cfg)
        if result and result['S'] < best_S_global:
            best_S_global = result['S']
            best_result_global = result

    if best_result_global is None:
        # Fallback
        pts = axial_to_cartesian(configs[0])
        center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        S = (max_proj + APOTHEM) / APOTHEM
        best_result_global = {'S': S, 'beta': beta, 'center': center, 'pts': pts}

    cx, cy = best_result_global['center']
    angle_deg = math.degrees(best_result_global['beta'])
    pts_final = best_result_global['pts']
    S_final = best_result_global['S']

    inner = []
    for x, y in pts_final:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)

    return inner, center, side, rotation
