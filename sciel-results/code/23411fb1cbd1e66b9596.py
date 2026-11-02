import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar
from itertools import product

N = 11
APOTHEM = math.sqrt(3) / 2.0
SQRT3 = math.sqrt(3)
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: all hexagons at origin (will overlap, but valid format)
        return [[0.0, 0.0, 0.0]] * N, [0.0, 0.0], 10.0, 0.0

    # Step 1: Generate many candidate topologies via axial coordinates
    # A unit hexagon at axial (q, r) has Cartesian center:
    #   x = sqrt(3) * (q + r/2), y = 1.5 * r
    # Two hexagons at (q1,r1) and (q2,r2) are disjoint if and only if
    #   max(|q1-q2|, |r1-r2|, |q1+r1-q2-r2|) >= 2
    # (since the hex grid spacing for side-1 hexagons is exactly 2 in all 3 axial directions)
    
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = SQRT3 * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def is_valid_packing(axial_list):
        """Check that all pairs of hexagons are disjoint (Chebyshev distance in axial >= 2)."""
        n = len(axial_list)
        for i in range(n):
            for j in range(i + 1, n):
                dq = axial_list[i][0] - axial_list[j][0]
                dr = axial_list[i][1] - axial_list[j][1]
                ds = (axial_list[i][0] + axial_list[i][1]) - (axial_list[j][0] + axial_list[j][1])
                if max(abs(dq), abs(dr), abs(ds)) < 2:
                    return False
        return True

    # Generate candidate topologies
    candidates = []

    # Classic compact shapes
    # Hexagon of side 2 minus 5 corners = 7 hexagons... need 11.
    # Let's generate systematically: all subsets of a grid that form valid packings

    # Approach: use a hex grid with axial coords in range, try all 11-subsets? Too many.
    # Instead, use known good shapes and perturbed versions.

    # Shape 1: 3x3 grid minus 2 corners
    shape1 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    if is_valid_packing(shape1):
        candidates.append(shape1)

    # Shape 2: elongated 2x5 with extras
    shape2 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (1, 2)
    ]
    if is_valid_packing(shape2):
        candidates.append(shape2)

    # Shape 3: 2x3 + 2x3 offset
    shape3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (3, 0), (4, 0), (5, 0),
        (3, 1), (4, 1)
    ]
    if is_valid_packing(shape3):
        candidates.append(shape3)

    # Shape 4: 3x4 minus 1
    shape4 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    if is_valid_packing(shape4):
        candidates.append(shape4)

    # Shape 5: 4x3 minus 1
    shape5 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3)
    ]
    if is_valid_packing(shape5):
        candidates.append(shape5)

    # Shape 6: 2x6 with one extra
    shape6 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (2, 2)
    ]
    if is_valid_packing(shape6):
        candidates.append(shape6)

    # Shape 7: 5x3 minus 4
    shape7 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    if is_valid_packing(shape7):
        candidates.append(shape7)

    # Shape 8: 3x5 minus 4
    shape8 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    if is_valid_packing(shape8):
        candidates.append(shape8)

    # Shape 9: 4x4 minus 5
    shape9 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3)
    ]
    if is_valid_packing(shape9):
        candidates.append(shape9)

    # Shape 10: 2x2 + 2x2 + 3 offset
    shape10 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (3, 0), (4, 0),
        (3, 1), (4, 1),
        (2, 2), (3, 2),
        (2, 3)
    ]
    if is_valid_packing(shape10):
        candidates.append(shape10)

    # Shape 11: 3x2 + 3x2 + 2 offset
    shape11 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (4, 0), (5, 0), (6, 0),
        (4, 1), (5, 1),
        (3, 2)
    ]
    if is_valid_packing(shape11):
        candidates.append(shape11)

    # Shape 12: 2x2 + 3x2 + 3 offset
    shape12 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (3, 0), (4, 0), (5, 0),
        (3, 1), (4, 1), (5, 1),
        (1, 3)
    ]
    if is_valid_packing(shape12):
        candidates.append(shape12)

    # Shape 13: 2x3 + 2x3 + 3 offset
    shape13 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (4, 0), (5, 0), (6, 0),
        (4, 1), (5, 1),
        (3, 2)
    ]
    if is_valid_packing(shape13):
        candidates.append(shape13)

    # Shape 14: 4x2 + 3 offset
    shape14 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2),
        (0, 3)
    ]
    if is_valid_packing(shape14):
        candidates.append(shape14)

    # Shape 15: 3x3 + 2 offset
    shape15 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (4, 0), (4, 1)
    ]
    if is_valid_packing(shape15):
        candidates.append(shape15)

    # Shape 16: 2x2 + 2x2 + 2x2 (three 2x2 blocks)
    shape16 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (3, 0), (4, 0),
        (3, 1), (4, 1),
        (6, 0), (7, 0),
        (6, 1)
    ]
    if is_valid_packing(shape16):
        candidates.append(shape16)

    # Shape 17: 3x2 + 2x2 + 3 offset
    shape17 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (4, 0), (5, 0),
        (4, 1), (5, 1),
        (3, 2)
    ]
    if is_valid_packing(shape17):
        candidates.append(shape17)

    # Shape 18: 2x3 + 2x3 + 2x2 offset
    shape18 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (4, 0), (5, 0), (6, 0),
        (4, 1), (5, 1),
        (3, 2), (3, 3)
    ]
    if is_valid_packing(shape18):
        candidates.append(shape18)

    # Shape 19: 3x3 + 2 offset (different)
    shape19 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    if is_valid_packing(shape19):
        candidates.append(shape19)

    # Shape 20: 4x3 - 1
    shape20 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (3, 2)
    ]
    if is_valid_packing(shape20):
        candidates.append(shape20)

    # Shape 21: 3x4 - 1
    shape21 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    if is_valid_packing(shape21):
        candidates.append(shape21)

    # Shape 22: 2x5 + 1
    shape22 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2)
    ]
    if is_valid_packing(shape22):
        candidates.append(shape22)

    # Shape 23: 5x2 + 1
    shape23 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    if is_valid_packing(shape23):
        candidates.append(shape23)

    # Shape 24: 3x3 + 2 offset (another variant)
    shape24 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 0), (3, 0)
    ]
    if is_valid_packing(shape24):
        candidates.append(shape24)

    # Shape 25: 4x2 + 2x2 offset
    shape25 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    if is_valid_packing(shape25):
        candidates.append(shape25)

    # Shape 26: 2x4 + 2x2 + 1
    shape26 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (5, 0), (6, 0),
        (5, 1), (6, 1),
        (4, 2)
    ]
    if is_valid_packing(shape26):
        candidates.append(shape26)

    # Shape 27: 3x2 + 2x3 + 2 offset
    shape27 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (4, 0), (5, 0),
        (4, 1), (5, 1), (6, 1),
        (3, 2)
    ]
    if is_valid_packing(shape27):
        candidates.append(shape27)

    # Shape 28: 2x3 + 3x2 + 2 offset
    shape28 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (4, 0), (4, 1), (5, 0), (5, 1),
        (6, 0), (6, 1),
        (3, 2)
    ]
    if is_valid_packing(shape28):
        candidates.append(shape28)

    # Shape 29: 3x3 + 2 offset (yet another)
    shape29 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 2)
    ]
    if is_valid_packing(shape29):
        candidates.append(shape29)

    # Shape 30: 2x2 + 3x2 + 3 offset (another)
    shape30 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (3, 0), (4, 0), (5, 0),
        (3, 1), (4, 1), (5, 1),
        (1, 2), (2, 2)
    ]
    if is_valid_packing(shape30):
        candidates.append(shape30)

    # Also add the original program's candidates that passed validation
    original_candidates = [
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)],
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)],
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(3,2),(1,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)],
    ]
    for oc in original_candidates:
        if is_valid_packing(oc) and oc not in candidates:
            candidates.append(oc)

    if not candidates:
        # Fallback: just put them in a line (will overlap but valid format)
        pts = np.array([(i * 2.0 * SQRT3, 0.0) for i in range(N)])
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        S = (max_proj + APOTHEM) / APOTHEM
        inner = [[float(x), float(y), 0.0] for x, y in pts]
        return inner, [float(cx), float(cy)], float(S), 0.0

    # Step 2: For each candidate, optimize the outer hexagon (center, rotation)
    # using LP + smooth surrogate + Nelder-Mead polish
    
    def calc_S_for_beta(pts, beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
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
                D = res.x[2]
                cx = res.x[0]
                cy = res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def smooth_calc_S(pts, beta, cx, cy, temperature=0.1):
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        max_val = np.max(all_projs)
        if max_val > 100:
            return float('inf')
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        S = (D + APOTHEM) / APOTHEM
        return S

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return best_S, best_beta, best_center
        
        # Coarse grid search over beta
        for i in range(37):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            beta = i * (math.pi / 6) / 36
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Refine beta
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0],
                                     bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 200, 'xatol': 1e-10})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass
        
        # Smooth surrogate optimization
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.2, 0.1, 0.05, 0.02]
                start_betas = [best_beta, best_beta + 0.05, best_beta - 0.05, 0.0, math.pi/12]
                
                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    for sb in start_betas:
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break
                        
                        def smooth_obj(params, t=temp):
                            cx, cy, beta = params
                            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                            return smooth_calc_S(pts, beta, cx, cy, temperature=t)
                        
                        init = [best_center[0], best_center[1], sb]
                        res = minimize(smooth_obj, init, method='Nelder-Mead',
                                     options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8})
                        
                        if res.fun < 1e10:
                            cx_r, cy_r, beta_r = res.x
                            beta_r = ((beta_r % (math.pi/6)) + math.pi/6) % (math.pi/6)
                            S_exact, c_exact, _ = calc_S_for_beta(pts, beta_r)
                            if S_exact < best_S:
                                best_S = S_exact
                                best_beta = beta_r
                                best_center = c_exact
        except Exception:
            pass
        
        # Final polish
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    normals = []
                    for k in range(6):
                        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                        normals.append((math.cos(ang), math.sin(ang)))
                    max_proj = 0.0
                    for nx, ny in normals:
                        for px, py in pts:
                            proj = abs((px - cx) * nx + (py - cy) * ny)
                            if proj > max_proj:
                                max_proj = proj
                    return (max_proj + APOTHEM) / APOTHEM
                
                polish_starts = [
                    [best_center[0], best_center[1], best_beta],
                    [best_center[0] + 0.02, best_center[1], best_beta],
                    [best_center[0] - 0.02, best_center[1], best_beta],
                    [best_center[0], best_center[1] + 0.02, best_beta],
                    [best_center[0], best_center[1] - 0.02, best_beta],
                ]
                
                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                 options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass
        
        return best_S, best_beta, best_center

    # Step 3: Evaluate all candidates and find the best
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for i, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(best_global_pts[:, 0].mean()), float(best_global_pts[:, 1].mean()))
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
    
    # Apply tiny safety scale
    SCALE = 1.0 + 1e-10
    pts_scaled = np.array([[cx + (x - cx) * SCALE, cy + (y - cy) * SCALE] for x, y in best_global_pts])
    
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
