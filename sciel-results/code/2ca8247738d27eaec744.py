import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar, linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def hex_dist_sq(p1, p2):
    """Squared distance between two points in Cartesian space."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def check_non_overlap(pts, threshold_sq=4.0 * (1 - 1e-6)):
    """Check if all pairs of points are at least 2 units apart."""
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            if hex_dist_sq(pts[i], pts[j]) < threshold_sq:
                return False
    return True

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Define candidate point sets (axial coordinates)
    candidates = []

    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    candidates.append(axial_1)

    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_2)

    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_3)

    axial_4 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_4)

    axial_5 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_5)

    axial_6 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_6)

    axial_7 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_7)

    axial_8 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append(axial_8)

    axial_9 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    candidates.append(axial_9)

    axial_10 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_10)

    axial_11 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)
    ]
    candidates.append(axial_11)

    axial_12 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_12)

    axial_13 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)
    ]
    candidates.append(axial_13)

    axial_14 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    candidates.append(axial_14)

    axial_15 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1), (0, -1)
    ]
    candidates.append(axial_15)

    axial_16 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_16)

    axial_17 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)
    ]
    candidates.append(axial_17)

    axial_18 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_18)

    axial_19 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_19)

    axial_20 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, 1)
    ]
    candidates.append(axial_20)

    axial_21 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_21)

    axial_22 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    candidates.append(axial_22)

    axial_23 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_23)

    axial_24 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_24)

    axial_25 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_25)

    axial_26 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_26)

    axial_27 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, -2)
    ]
    candidates.append(axial_27)

    axial_28 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_28)

    axial_29 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, -1), (2, -1)
    ]
    candidates.append(axial_29)

    axial_30 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3),
        (1, -1)
    ]
    candidates.append(axial_30)

    axial_31 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3),
        (1, -1), (2, -1)
    ]
    candidates.append(axial_31)

    axial_32 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, -2)
    ]
    candidates.append(axial_32)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def calc_S_lp(pts, beta):
        """Use LP to find optimal center and D for fixed beta."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        # Variables: cx, cy, D
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for px, py in pts:
                # n · (p - c) <= D  =>  -nx*cx - ny*cy - D <= -n·p
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                # n · (c - p) <= D  =>  nx*cx + ny*cy - D <= n·p
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

    def calc_S_smooth(pts, beta):
        """Use smooth optimization with log-sum-exp surrogate for fixed beta."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        
        normals = np.array([
            [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
            for k in range(6)
        ])
        
        def D_of_center(cx, cy):
            c = np.array([cx, cy])
            nc = normals @ c
            projs = pts @ normals.T
            a = np.max(projs, axis=0)
            b = np.min(projs, axis=0)
            # Use soft-max for smoothness
            vals = np.maximum(a - nc, nc - b)
            # Log-sum-exp approximation of max
            max_val = np.max(vals)
            if max_val > 1e6:
                return max_val
            log_sum = max_val + math.log(np.sum(np.exp(vals - max_val)))
            return log_sum
        
        cx_center = (np.max(pts[:, 0]) + np.min(pts[:, 0])) / 2
        cy_center = (np.max(pts[:, 1]) + np.min(pts[:, 1])) / 2
        
        best_D = float('inf')
        best_c = (cx_center, cy_center)
        
        cx_range = np.max(pts[:, 0]) - np.min(pts[:, 0])
        cy_range = np.max(pts[:, 1]) - np.min(pts[:, 1])
        
        for dx in np.linspace(-cx_range*0.3, cx_range*0.3, 5):
            for dy in np.linspace(-cy_range*0.3, cy_range*0.3, 5):
                D = D_of_center(cx_center + dx, cy_center + dy)
                if D < best_D:
                    best_D = D
                    best_c = (cx_center + dx, cy_center + dy)
        
        try:
            res = minimize(D_of_center, best_c, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_D:
                best_D = res.fun
                best_c = (res.x[0], res.x[1])
        except Exception:
            pass
        
        S = (best_D + APOTHEM) / APOTHEM
        return S, best_c, best_D

    def calc_S_nm(pts, beta):
        """Use Nelder-Mead as fallback for fixed beta."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        
        normals = np.array([
            [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
            for k in range(6)
        ])
        
        projs = pts @ normals.T
        a = np.max(projs, axis=0)
        b = np.min(projs, axis=0)
        
        def D_of_center(cx, cy):
            c = np.array([cx, cy])
            nc = normals @ c
            vals = np.maximum(a - nc, nc - b)
            return np.max(vals)
        
        cx_center = (np.max(pts[:, 0]) + np.min(pts[:, 0])) / 2
        cy_center = (np.max(pts[:, 1]) + np.min(pts[:, 1])) / 2
        
        best_D = float('inf')
        best_c = (cx_center, cy_center)
        
        cx_range = np.max(pts[:, 0]) - np.min(pts[:, 0])
        cy_range = np.max(pts[:, 1]) - np.min(pts[:, 1])
        
        for dx in np.linspace(-cx_range*0.3, cx_range*0.3, 5):
            for dy in np.linspace(-cy_range*0.3, cy_range*0.3, 5):
                D = D_of_center(cx_center + dx, cy_center + dy)
                if D < best_D:
                    best_D = D
                    best_c = (cx_center + dx, cy_center + dy)
        
        try:
            res = minimize(D_of_center, best_c, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_D:
                best_D = res.fun
                best_c = (res.x[0], res.x[1])
        except Exception:
            pass
        
        S = (best_D + APOTHEM) / APOTHEM
        return S, best_c, best_D

    def calc_S(pts, beta):
        """Hybrid: try LP first, then smooth, then NM."""
        S_lp, c_lp, D_lp = calc_S_lp(pts, beta)
        if S_lp < float('inf'):
            return S_lp, c_lp, D_lp
        S_smooth, c_smooth, D_smooth = calc_S_smooth(pts, beta)
        S_nm, c_nm, D_nm = calc_S_nm(pts, beta)
        if S_smooth < S_nm:
            return S_smooth, c_smooth, D_smooth
        return S_nm, c_nm, D_nm

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None
        
        # Coarse grid search over beta
        for i in range(13):
            beta = i * (math.pi / 6) / 12
            S, center, D = calc_S(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Refine beta with bounded scalar optimization
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S(pts, b)[0], bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 200, 'xatol': 1e-10})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S(pts, best_beta)
        except Exception:
            pass
        
        # Final high-precision polish using alternating optimization
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    normals = np.array([
                        [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
                         math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
                        for k in range(6)
                    ])
                    projs = (pts - np.array([cx, cy])) @ normals.T
                    max_proj = np.max(np.abs(projs))
                    S = (max_proj + APOTHEM) / APOTHEM
                    return S
                
                cx, cy = best_center
                beta = best_beta
                
                for _ in range(8):
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    
                    # Optimize cx, cy for fixed beta
                    def obj_c(params):
                        return objective([params[0], params[1], beta])
                    
                    res_c = minimize(obj_c, [cx, cy], method='Nelder-Mead',
                                     options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
                    cx, cy = res_c.x
                    S_c = res_c.fun
                    
                    # Optimize beta for fixed cx, cy
                    def obj_b(params):
                        return objective([cx, cy, params[0]])
                    
                    b_starts = np.linspace(0, math.pi/6, 25)
                    best_b = beta
                    best_S_b = S_c
                    for b0 in b_starts:
                        if abs(b0 - best_b) < 1e-3: continue
                        res_b = minimize_scalar(obj_b, bounds=(0, math.pi/6), method='bounded',
                                                options={'maxiter': 100, 'xatol': 1e-10})
                        if res_b.fun < best_S_b:
                            best_S_b = res_b.fun
                            best_b = res_b.x
                    beta = best_b
                    S_b = best_S_b
                    
                    if S_b < best_S:
                        best_S = S_b
                        best_center = (cx, cy)
                        best_beta = beta
                        
        except Exception:
            pass
            
        return best_S, best_beta, best_center

    def lattice_search_for_config(axial_coords, time_budget=10.0):
        """Search for non-overlapping positions on a fine lattice."""
        start_time = time.time()
        base_pts = axial_to_cartesian(axial_coords)
        
        # Try small perturbations on a fine lattice
        best_S = float('inf')
        best_positions = None
        best_beta = 0.0
        best_center = None
        
        # Grid of perturbation offsets
        offsets = []
        for dx in np.linspace(-0.3, 0.3, 7):
            for dy in np.linspace(-0.3, 0.3, 7):
                offsets.append((dx, dy))
        
        # For each offset pattern, check if we can assign offsets to avoid overlap
        # This is a simplified approach: try uniform scaling of the whole config
        for scale in np.linspace(0.95, 1.05, 21):
            if time.time() - start_time > time_budget:
                break
            scaled_pts = base_pts * scale
            # Check non-overlap
            if check_non_overlap(scaled_pts, 4.0 * (1 - 1e-6)):
                # Find optimal container for this configuration
                for beta in np.linspace(0, math.pi/6, 13):
                    if time.time() - start_time > time_budget:
                        break
                    S, center, D = calc_S(scaled_pts, beta)
                    if S < best_S:
                        best_S = S
                        best_positions = scaled_pts
                        best_beta = beta
                        best_center = center
        
        if best_positions is not None:
            return best_S, best_positions, best_beta, best_center
        return float('inf'), None, 0.0, None

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    # First pass: optimize each candidate configuration
    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center
    
    # Second pass: try lattice search with scaling for top candidates
    remaining_time = TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)
    if remaining_time > 15.0:
        # Sort candidates by their best_S from first pass and try lattice search on best ones
        candidate_results = []
        for axial in candidates:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            pts = axial_to_cartesian(axial)
            S, beta, center = calc_S(pts, 0.0)  # Quick estimate
            candidate_results.append((S, axial))
        
        candidate_results.sort(key=lambda x: x[0])
        
        for S_est, axial in candidate_results[:5]:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            S_lattice, pts_lattice, beta_lattice, center_lattice = lattice_search_for_config(axial, time_budget=min(15.0, remaining_time / 2))
            if S_lattice < best_global_S:
                best_global_S = S_lattice
                best_global_pts = pts_lattice
                best_global_beta = beta_lattice
                best_global_center = center_lattice

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
    
    # Use a very small safety margin to ensure no overlap
    SCALE = 1.0 + 1e-12
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    
    pts_scaled = np.array(pts_scaled)
    
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
