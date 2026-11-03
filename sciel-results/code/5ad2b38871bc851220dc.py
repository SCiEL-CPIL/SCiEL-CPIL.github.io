import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar
import sys

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
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

    # Generate diverse candidate configurations
    candidates = []

    # Standard hexagonal cluster
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)])
    # Cross
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    # 3x3 block + 2
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)])
    # 4x3 minus corners
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)])
    # Staircase
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    # Compact 3x3+2
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    # Star
    candidates.append([(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)])
    # Elongated
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)])
    # Asymmetric star
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)])
    # Compact with top
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    # Wide base
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)])
    # Tall narrow
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)])
    # Star with asymmetric outer
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)])
    # Star with diagonal
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
    # Dense cluster
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)])
    # Zigzag
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)])
    # Star with top cluster
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)])
    # Trapezoid
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)])
    # Wide top
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)])
    # Star with inner diagonal
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)])
    # 4x3 variant
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    # Wide with top
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)])
    # Symmetric star
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    # Very tall
    candidates.append([(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)])
    # 3x3+row
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    # Dense 4-wide
    candidates.append([(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)])
    # Star with asymmetric bottom
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)])
    # Tall with top row
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)])
    # Additional diverse shapes
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(0,-1),(3,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(1,4),(2,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3)])
    # New: more compact arrangements
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(0,-1)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2),(0,-1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(3,2),(0,3)])

    def exact_S(pts, cx, cy, beta):
        """Exact computation of S for given center and rotation."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        D = np.max(np.abs(projs))
        return (D + APOTHEM) / APOTHEM

    def check_overlap(pts, min_dist=2.0*APOTHEM*0.9999):
        """Check if any two hexagon centers are too close (overlap)."""
        n = len(pts)
        # Vectorized distance check is faster for small n, but loop is fine for N=11
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_dist - 1e-7:
                    return False
        return True

    def smooth_objective(params, pts_init, temperature):
        """
        Smooth surrogate for the minimum enclosing hexagon size.
        params: [x1, y1, ..., xn, yn, cx, cy, beta]
        Returns: Approximate S + collision penalty
        """
        n = len(pts_init)
        pts = params[:2*n].reshape(n, 2)
        cx, cy = params[2*n], params[2*n+1]
        beta = params[2*n+2]
        
        # Periodic boundary for beta (0 to pi/6)
        beta_mod = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)

        # Collision penalty
        min_dist = 2.0 * APOTHEM
        pen = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_dist:
                    pen += 50.0 * (min_dist - d) ** 2
        if pen > 5.0:
            return pen + 100.0

        # Smooth Min-Enclosing Hexagon
        angles = beta_mod + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        pts_centered = pts - np.array([cx, cy])
        # projs[i, k] is projection of point i on normal k
        projs = pts_centered @ normals.T 
        
        # We want max over all points and all signs of normals
        # Equivalent to max of |projs|
        # Use log-sum-exp for max
        # D = max_{i,k} |proj_{i,k}|
        all_projs = np.abs(projs).ravel()
        
        # Stability for log-sum-exp
        max_val = np.max(all_projs)
        if max_val > 100:
            return max_val + 100.0
            
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D_smooth = temperature * log_sum
        
        S_approx = (D_smooth + APOTHEM) / APOTHEM
        
        return S_approx + 0.1 * pen

    def optimize_continuous(pts_init, axial):
        """
        Optimizes positions and outer hexagon parameters simultaneously.
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return None, None, None, None

        n = len(pts_init)
        # Initial guess for center: centroid
        cx0 = np.mean(pts_init[:, 0])
        cy0 = np.mean(pts_init[:, 1])
        
        # Initial guess for beta: try a few values
        best_S = float('inf')
        best_params = None
        best_pts = None
        best_center = None
        best_beta = None

        # First, find a good starting beta with fixed positions
        for beta_try in [0.0, math.pi/12, math.pi/6]:
            S, _, _ = calc_S_for_beta(pts_init, beta_try)
            if S < best_S:
                best_S = S
                best_beta = beta_try
                # Get center from LP
                angles = beta_try + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
                normals = np.column_stack([np.cos(angles), np.sin(angles)])
                # Simple centering: midpoint of bounding box in all directions
                # For better init, we can just use centroid for now, LP will refine
                cx0 = np.mean(pts_init[:, 0])
                cy0 = np.mean(pts_init[:, 1])

        # Now run smooth optimization
        # Parameters: x1...xn, y1...yn, cx, cy, beta
        x0 = np.concatenate([pts_init.ravel(), [cx0, cy0, best_beta]])
        
        # Try different temperatures for annealing-like effect
        temperatures = [0.5, 0.2, 0.1, 0.05, 0.02]
        
        for temp in temperatures:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            try:
                res = minimize(
                    smooth_objective, 
                    x0, 
                    args=(pts_init, temp), 
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                if res.fun < 10.0: # Sanity check
                    res_x = res.x
                    res_pts = res_x[:2*n].reshape(n, 2)
                    res_cx, res_cy, res_beta = res_x[2*n], res_x[2*n+1], res_x[2*n+2]
                    res_beta = ((res_beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    
                    if check_overlap(res_pts):
                        S_exact = exact_S(res_pts, res_cx, res_cy, res_beta)
                        if S_exact < best_S:
                            best_S = S_exact
                            best_params = res_x
                            best_pts = res_pts
                            best_center = (res_cx, res_cy)
                            best_beta = res_beta
            except Exception:
                pass

        return best_S, best_pts, best_center, best_beta

    def calc_S_for_beta(pts, beta):
        """Vectorized LP for minimum enclosing hexagon with given rotation."""
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

    def refine_lp(pts, beta, cx_init, cy_init):
        """Refine center and size using LP with fixed points and rotation."""
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
                return S, (cx, cy)
        except:
            pass
        return float('inf'), None

    # Main loop
    best_global_S = float('inf')
    best_global_pts = None
    best_global_center = None
    best_global_beta = 0.0

    # Sort candidates by initial static score to prioritize good ones
    scored_candidates = []
    for axial in candidates:
        pts = axial_to_cartesian(axial)
        S, center, _ = calc_S_for_beta(pts, 0.0) # Quick static check at beta=0
        scored_candidates.append((S, axial, pts))
    scored_candidates.sort(key=lambda x: x[0])

    for idx, (static_S, axial, pts) in enumerate(scored_candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # 1. Try continuous optimization
        S_cont, pts_opt, center_opt, beta_opt = optimize_continuous(pts, axial)
        
        if pts_opt is not None and S_cont < best_global_S:
            # 2. Refine with exact LP for the final positions
            S_final, center_final = refine_lp(pts_opt, beta_opt, center_opt[0], center_opt[1])
            if S_final < best_global_S:
                best_global_S = S_final
                best_global_pts = pts_opt
                best_global_center = center_final
                best_global_beta = beta_opt

    # Fallback if no continuous optimization succeeded
    if best_global_center is None or best_global_S == float('inf'):
        # Use the best static candidate
        best_axial = scored_candidates[0][1]
        best_global_pts = axial_to_cartesian(best_axial)
        S, center, _ = calc_S_for_beta(best_global_pts, 0.0)
        best_global_S = S
        best_global_center = center
        best_global_beta = 0.0

    if best_global_center is None:
        # Ultimate fallback
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

    # Final scaling for safety (very tiny)
    SCALE = 1.0 + 1e-15
    pts_scaled = best_global_pts.copy()
    for i in range(len(pts_scaled)):
        pts_scaled[i, 0] = cx + (pts_scaled[i, 0] - cx) * SCALE
        pts_scaled[i, 1] = cy + (pts_scaled[i, 1] - cy) * SCALE

    # Recalculate final side length
    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))

    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM

    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)

    return inner, center, side, rotation

