import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar, linprog
from scipy.spatial import Delaunay

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    # --- Candidate point sets (axial coordinates) ---
    candidates = []
    
    # Hexagonal cluster: center + 6 neighbors + 4 more
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    candidates.append(axial_1)
    
    # Star: center + 6 neighbors + 4 opposite
    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_2)
    
    # 3x3 block + 2 extras
    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_3)
    
    # 4x3 block + 1 extra
    axial_4 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_4)
    
    # 3x4 block + 1 extra
    axial_5 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_5)
    
    # 3x3 + 2
    axial_6 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_6)
    
    # Cross-like
    axial_7 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_7)
    
    # 5+6 row
    axial_8 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append(axial_8)
    
    # Center + 6 + 4 asymmetric
    axial_9 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    candidates.append(axial_9)
    
    # 3x3 + 2 top
    axial_10 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_10)
    
    # 2+4+4+1
    axial_11 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)
    ]
    candidates.append(axial_11)
    
    # 4+2+2+2+1
    axial_12 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_12)
    
    # Center + 6 + 4 (variant)
    axial_13 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)
    ]
    candidates.append(axial_13)
    
    # Center + 6 + 4 (variant)
    axial_14 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    candidates.append(axial_14)
    
    # 3x3 with extras
    axial_15 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1), (0, -1)
    ]
    candidates.append(axial_15)
    
    # Staircase
    axial_16 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_16)
    
    # Center + 6 + 4 (variant)
    axial_17 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)
    ]
    candidates.append(axial_17)
    
    # 4+3+3
    axial_18 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_18)
    
    # 2+3+2+3+1
    axial_19 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_19)
    
    # Center + 6 + 4 (variant)
    axial_20 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, 1)
    ]
    candidates.append(axial_20)
    
    # 4+4+3
    axial_21 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_21)
    
    # 3+4+3+1
    axial_22 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    candidates.append(axial_22)
    
    # Center + 6 + 4 (variant)
    axial_23 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_23)
    
    # 2+2+3+2+2
    axial_24 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_24)
    
    # 3+2+3+3
    axial_25 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_25)
    
    # Compact: 2+3+3+2+1
    axial_36 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (1, 4)
    ]
    candidates.append(axial_36)
    
    # Compact: 3+3+3+1+1
    axial_37 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3),
        (0, 3)
    ]
    candidates.append(axial_37)
    
    # Compact: 4+2+2+2+1 (alternate)
    axial_38 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_38)
    
    # Compact: 2+4+3+1+1
    axial_39 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3),
        (2, 3)
    ]
    candidates.append(axial_39)
    
    # Compact: 3+4+2+1+1
    axial_40 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2),
        (0, 3),
        (3, 3)
    ]
    candidates.append(axial_40)
    
    # Compact: 2+2+4+2+1
    axial_41 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3),
        (2, 3)
    ]
    candidates.append(axial_41)
    
    # Compact: 3+3+2+2+1
    axial_42 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3),
        (2, 3)
    ]
    candidates.append(axial_42)
    
    # Compact: 4+3+2+1+1
    axial_43 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3),
        (3, 3)
    ]
    candidates.append(axial_43)
    
    # Compact: 2+3+3+2+1 (alternate)
    axial_44 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_44)
    
    # 4+3+4+2
    axial_26 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_26)
    
    # Center + 6 + 4 (variant)
    axial_27 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, -2)
    ]
    candidates.append(axial_27)
    
    # 2+3+2+3+3
    axial_28 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_28)
    
    # Additional diverse candidates
    axial_29 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_29)
    
    axial_30 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)
    ]
    candidates.append(axial_30)
    
    axial_31 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_31)
    
    axial_32 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_32)
    
    axial_33 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_33)
    
    axial_34 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_34)
    
    axial_35 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_35)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def calc_S_for_beta(pts, beta, cx, cy):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf')
        
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        centered = pts - np.array([cx, cy])
        projs = centered @ normals.T
        max_proj = np.max(np.abs(projs))
        return (max_proj + APOTHEM) / APOTHEM

    def fast_exact_S(pts, cx, cy, beta):
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        centered = pts - np.array([cx, cy])
        projs = centered @ normals.T
        max_proj = np.max(np.abs(projs))
        return (max_proj + APOTHEM) / APOTHEM

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

    def voronoi_constraint_check(pts, min_dist=1.99):
        """Check if all pairs of points are at least min_dist apart."""
        n = len(pts)
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_dist:
                    return False
        return True

    def optimize_with_voronoi(pts, current_best_S=float('inf')):
        """
        Optimize positions using Voronoi-based distance constraints.
        Key insight: minimize the maximum distance from any point to the outer hexagon boundary,
        while maintaining minimum inter-point distance >= 2 (for non-overlapping unit hexagons).
        """
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        best_pts = pts.copy()
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None, None
        
        # Phase 1: Coarse grid search over beta using LP
        for i in range(37):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                return best_S, best_beta, best_center, best_pts
            beta = i * (math.pi / 6) / 36
            
            # Compute optimal center for this beta
            angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            
            n_pts = len(pts)
            A_ub_list = []
            b_ub_list = []
            for k in range(6):
                nx, ny = normals[k]
                projs = pts[:, 0] * nx + pts[:, 1] * ny
                A_ub_list.append(np.column_stack([-np.full(n_pts, nx), -np.full(n_pts, ny), -np.ones(n_pts)]))
                b_ub_list.append(-projs)
                A_ub_list.append(np.column_stack([np.full(n_pts, nx), np.full(n_pts, ny), -np.ones(n_pts)]))
                b_ub_list.append(projs)
            
            A_ub = np.vstack(A_ub_list)
            b_ub = np.concatenate(b_ub_list)
            c_obj = np.array([0.0, 0.0, 1.0])
            bounds = [(None, None), (None, None), (0, None)]
            
            try:
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if res.status == 0:
                    D = res.x[2]
                    cx = res.x[0]
                    cy = res.x[1]
                    S = (D + APOTHEM) / APOTHEM
                    if S < best_S:
                        best_S = S
                        best_beta = beta
                        best_center = (cx, cy)
            except Exception:
                pass
        
        # Early exit
        if best_S >= current_best_S * 1.001 and current_best_S < float('inf'):
            return best_S, best_beta, best_center, best_pts
        
        # Phase 2: Refine beta
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def obj_beta(b):
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        return float('inf')
                    # Quick center estimate
                    cx = np.mean(pts[:, 0])
                    cy = np.mean(pts[:, 1])
                    return calc_S_for_beta(pts, b, cx, cy)
                
                res = minimize_scalar(obj_beta, bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 200, 'xatol': 1e-10})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    cx = np.mean(pts[:, 0])
                    cy = np.mean(pts[:, 1])
                    best_center = (cx, cy)
        except Exception:
            pass
        
        # Phase 3: Position optimization with Voronoi constraints
        # Key: allow positions to move, but enforce minimum distance between centers
        # This is the structural change: dynamic position optimization
        
        if best_center is not None:
            bcx, bcy = best_center
            bb = best_beta
            
            # Objective: maximize the minimum clearance from outer boundary
            # Equivalent to: minimize max over all points of (projection onto normals)
            def objective_positions(params):
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    return float('inf')
                
                # params: flattened array of 22 values = 11 (x,y) pairs
                pts_opt = params.reshape(-1, 2)
                
                # Voronoi distance constraint: all pairs must be >= 1.99 apart
                n = len(pts_opt)
                min_dist = 1.99
                for i in range(n):
                    for j in range(i+1, n):
                        d = np.linalg.norm(pts_opt[i] - pts_opt[j])
                        if d < min_dist:
                            # Penalty for violation
                            penalty = (min_dist - d) * 100.0
                            return penalty
                
                # Compute S
                angles = bb + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
                normals = np.column_stack([np.cos(angles), np.sin(angles)])
                centered = pts_opt - np.array([bcx, bcy])
                projs = centered @ normals.T
                max_proj = np.max(np.abs(projs))
                return (max_proj + APOTHEM) / APOTHEM
            
            # Initial guess: original positions
            init = pts.flatten()
            
            # Multiple restarts with small perturbations
            for trial in range(3):
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                
                noise = 0.02 * np.random.randn(len(init))
                init_noisy = init + noise
                
                try:
                    res = minimize(objective_positions, init_noisy, method='Nelder-Mead',
                                 options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12})
                    
                    if res.fun < best_S:
                        pts_opt = res.x.reshape(-1, 2)
                        if voronoi_constraint_check(pts_opt):
                            # Recompute optimal center
                            angles = bb + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
                            normals = np.column_stack([np.cos(angles), np.sin(angles)])
                            
                            n_pts = len(pts_opt)
                            A_ub_list = []
                            b_ub_list = []
                            for k in range(6):
                                nx, ny = normals[k]
                                projs = pts_opt[:, 0] * nx + pts_opt[:, 1] * ny
                                A_ub_list.append(np.column_stack([-np.full(n_pts, nx), -np.full(n_pts, ny), -np.ones(n_pts)]))
                                b_ub_list.append(-projs)
                                A_ub_list.append(np.column_stack([np.full(n_pts, nx), np.full(n_pts, ny), -np.ones(n_pts)]))
                                b_ub_list.append(projs)
                            
                            A_ub = np.vstack(A_ub_list)
                            b_ub = np.concatenate(b_ub_list)
                            c_obj = np.array([0.0, 0.0, 1.0])
                            bounds = [(None, None), (None, None), (0, None)]
                            
                            res2 = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                            if res2.status == 0:
                                D = res2.x[2]
                                cx = res2.x[0]
                                cy = res2.x[1]
                                S = (D + APOTHEM) / APOTHEM
                                if S < best_S:
                                    best_S = S
                                    best_pts = pts_opt
                                    best_center = (cx, cy)
                except Exception:
                    pass
        
        # Phase 4: Polish center and rotation
        if best_center is not None:
            bcx, bcy = best_center
            bb = best_beta
            pts_use = best_pts
            
            def objective_polish(params):
                cx, cy, beta = params
                beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                return fast_exact_S(pts_use, cx, cy, beta)
            
            polish_starts = [
                [bcx, bcy, bb],
                [bcx, bcy, bb + 0.02],
                [bcx, bcy, bb - 0.02],
                [bcx + 0.02, bcy, bb],
                [bcx - 0.02, bcy, bb],
                [bcx, bcy + 0.02, bb],
                [bcx, bcy - 0.02, bb],
            ]
            
            for init in polish_starts:
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                try:
                    res = minimize(objective_polish, init, method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                except Exception:
                    pass
        
        return best_S, best_beta, best_center, best_pts

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center, pts_opt = optimize_with_voronoi(pts, current_best_S=best_global_S)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts_opt if pts_opt is not None else pts
            best_global_beta = beta
            best_global_center = center

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
