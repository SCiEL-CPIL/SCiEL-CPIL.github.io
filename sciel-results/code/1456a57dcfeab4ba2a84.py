import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

# Precompute normals for overlap check (fixed orientation relative to each other)
# For two unit hexagons with same orientation, non-overlap condition is:
# min_k |d . n_k| >= sqrt(3)
# where n_k are the 6 normals of the hexagon.
NORMALS = np.array([
    [math.cos(math.pi / 6 + k * math.pi / 3), math.sin(math.pi / 6 + k * math.pi / 3)]
    for k in range(6)
])

def axial_to_cartesian(axial_list):
    pts = []
    for q, r in axial_list:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    return np.array(pts)

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Candidate seed configurations
    candidates = [
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 0), (-2, 0), (0, 2), (1, -2)],
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (1, 3), (2, 3)],
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 1), (1, 2), (-1, 2), (0, 3)],
        [(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (1, 3), (2, 3), (0, -1)],
        [(0, 0), (1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (1, 3), (2, 3)],
        [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2), (1, 3), (2, 3), (0, 4), (1, 4)],
    ]

    best_S = float('inf')
    best_pts = None
    best_center = None
    best_beta = 0.0

    def get_objective_and_constraints(pts, cx, cy, beta):
        # Objective: minimize outer side length S
        # S = (D + APOTHEM) / APOTHEM
        # D = max_k max_i | (p_i - c) . n_k(beta) |
        # n_k(beta) are normals of outer hexagon rotated by beta
        
        outer_normals = np.array([
            [math.cos(beta + math.pi / 6 + k * math.pi / 3),
             math.sin(beta + math.pi / 6 + k * math.pi / 3)]
            for k in range(6)
        ])
        
        centered = pts - np.array([cx, cy])
        projs = centered @ outer_normals.T
        D = np.max(np.abs(projs))
        S = (D + APOTHEM) / APOTHEM
        
        # Constraints: non-overlap
        # For each pair (i, j), min_k | (p_i - p_j) . n_k | >= sqrt(3)
        # We use a smooth approximation or direct check.
        # For SLSQP, we need smooth constraints.
        # Let g_ij_k = | (p_i - p_j) . n_k | - sqrt(3) >= 0
        # This is not smooth at 0.
        # Better: use squared distance in the "hexagonal metric" or similar.
        # Or simply use the raw difference and let SLSQP handle it, or use a penalty.
        # Let's use a penalty method for robustness in the objective, 
        # and keep the constraints as simple bounds if possible, or just penalty.
        
        # Actually, SLSQP works best with explicit constraints.
        # Let's define constraints as functions returning array of constraint values (>=0).
        # For each pair i<j and each normal k:
        # c = (p_i - p_j) . n_k - sqrt(3)  (one side)
        # c = -((p_i - p_j) . n_k) - sqrt(3) (other side)
        # This is 2 * N*(N-1)/2 * 6 constraints. Too many? 55 pairs * 6 * 2 = 660 constraints.
        # SLSQP might struggle.
        
        # Alternative: Penalty method.
        # Objective = S + lambda * sum(overlap_violations^2)
        
        return S, D

    def solve_packing(pts_init, beta_init, cx_init, cy_init):
        """
        Optimize positions, center, and rotation to minimize S.
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None, None

        n_pts = len(pts_init)
        
        def objective(params):
            pts = params[:2*n_pts].reshape(n_pts, 2)
            cx = params[2*n_pts]
            cy = params[2*n_pts+1]
            beta = params[2*n_pts+2]
            
            # Normalize beta
            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
            
            outer_normals = np.array([
                [math.cos(beta + math.pi / 6 + k * math.pi / 3),
                 math.sin(beta + math.pi / 6 + k * math.pi / 3)]
                for k in range(6)
            ])
            
            centered = pts - np.array([cx, cy])
            projs = centered @ outer_normals.T
            D = np.max(np.abs(projs))
            S = (D + APOTHEM) / APOTHEM
            
            # Overlap penalty
            penalty = 0.0
            for i in range(n_pts):
                for j in range(i+1, n_pts):
                    d = pts[i] - pts[j]
                    projs_overlap = np.abs(d @ NORMALS.T)
                    min_proj = np.min(projs_overlap)
                    if min_proj < math.sqrt(3):
                        violation = math.sqrt(3) - min_proj
                        penalty += violation * violation * 1000.0
            
            return S + penalty
        
        # Initial guess
        init_params = np.concatenate([pts_init.ravel(), [cx_init, cy_init, beta_init]])
        
        # Use SLSQP for local optimization
        try:
            res = minimize(objective, init_params, method='SLSQP',
                           options={'maxiter': 1000, 'ftol': 1e-12})
            
            if res.success:
                pts_opt = res.x[:2*n_pts].reshape(n_pts, 2)
                cx_opt = res.x[2*n_pts]
                cy_opt = res.x[2*n_pts+1]
                beta_opt = ((res.x[2*n_pts+2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                
                # Verify no overlap
                valid = True
                for i in range(n_pts):
                    for j in range(i+1, n_pts):
                        d = pts_opt[i] - pts_opt[j]
                        projs_overlap = np.abs(d @ NORMALS.T)
                        if np.min(projs_overlap) < math.sqrt(3) - 1e-7:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Recalculate exact S
                    outer_normals = np.array([
                        [math.cos(beta_opt + math.pi / 6 + k * math.pi / 3),
                         math.sin(beta_opt + math.pi / 6 + k * math.pi / 3)]
                        for k in range(6)
                    ])
                    centered = pts_opt - np.array([cx_opt, cy_opt])
                    projs = centered @ outer_normals.T
                    D = np.max(np.abs(projs))
                    S_exact = (D + APOTHEM) / APOTHEM
                    return S_exact, pts_opt, (cx_opt, cy_opt), beta_opt
        except Exception:
            pass
        
        return float('inf'), None, None, None

    # Main loop: Try each candidate, with some random perturbations (Simulated Annealing lite)
    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        pts = axial_to_cartesian(axial)
        
        # Initial center and beta
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        beta = 0.0
        
        # Try a few random perturbations to escape local minima
        for restart in range(3):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            # Perturb positions slightly
            if restart > 0:
                np.random.seed(int(time.time() * 1000) + restart)
                perturbation = np.random.normal(0, 0.1, (len(pts), 2))
                pts_perturbed = pts + perturbation
            else:
                pts_perturbed = pts
            
            S, pts_opt, center_opt, beta_opt = solve_packing(pts_perturbed, beta, cx, cy)
            
            if S < best_S:
                best_S = S
                best_pts = pts_opt
                best_center = center_opt
                best_beta = beta_opt
        
        # If we found a very good solution, break early
        if best_S < 3.2:
            break

    # Fallback
    if best_pts is None:
        pts = axial_to_cartesian(candidates[0])
        best_pts = pts
        best_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        best_beta = 0.0
        outer_normals = np.array([
            [math.cos(best_beta + math.pi / 6 + k * math.pi / 3),
             math.sin(best_beta + math.pi / 6 + k * math.pi / 3)]
            for k in range(6)
        ])
        centered = best_pts - np.array(best_center)
        projs = centered @ outer_normals.T
        D = np.max(np.abs(projs))
        best_S = (D + APOTHEM) / APOTHEM

    # Final polish: try to tighten further with a very small step
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        n_pts = len(best_pts)
        def final_obj(params):
            pts = params[:2*n_pts].reshape(n_pts, 2)
            cx = params[2*n_pts]
            cy = params[2*n_pts+1]
            beta = params[2*n_pts+2]
            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
            
            outer_normals = np.array([
                [math.cos(beta + math.pi / 6 + k * math.pi / 3),
                 math.sin(beta + math.pi / 6 + k * math.pi / 3)]
                for k in range(6)
            ])
            
            centered = pts - np.array([cx, cy])
            projs = centered @ outer_normals.T
            D = np.max(np.abs(projs))
            S = (D + APOTHEM) / APOTHEM
            
            penalty = 0.0
            for i in range(n_pts):
                for j in range(i+1, n_pts):
                    d = pts[i] - pts[j]
                    projs_overlap = np.abs(d @ NORMALS.T)
                    min_proj = np.min(projs_overlap)
                    if min_proj < math.sqrt(3):
                        violation = math.sqrt(3) - min_proj
                        penalty += violation * violation * 10000.0
            
            return S + penalty

        init_final = np.concatenate([best_pts.ravel(), [best_center[0], best_center[1], best_beta]])
        try:
            res_final = minimize(final_obj, init_final, method='SLSQP',
                                 options={'maxiter': 500, 'ftol': 1e-14})
            if res_final.success:
                pts_f = res_final.x[:2*n_pts].reshape(n_pts, 2)
                cx_f = res_final.x[2*n_pts]
                cy_f = res_final.x[2*n_pts+1]
                beta_f = ((res_final.x[2*n_pts+2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                
                # Check validity
                valid = True
                for i in range(n_pts):
                    for j in range(i+1, n_pts):
                        d = pts_f[i] - pts_f[j]
                        projs_overlap = np.abs(d @ NORMALS.T)
                        if np.min(projs_overlap) < math.sqrt(3) - 1e-7:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    outer_normals = np.array([
                        [math.cos(beta_f + math.pi / 6 + k * math.pi / 3),
                         math.sin(beta_f + math.pi / 6 + k * math.pi / 3)]
                        for k in range(6)
                    ])
                    centered = pts_f - np.array([cx_f, cy_f])
                    projs = centered @ outer_normals.T
                    D = np.max(np.abs(projs))
                    S_final = (D + APOTHEM) / APOTHEM
                    if S_final < best_S:
                        best_S = S_final
                        best_pts = pts_f
                        best_center = (cx_f, cy_f)
                        best_beta = beta_f
        except Exception:
            pass

    cx, cy = best_center
    angle_deg = math.degrees(best_beta)
    
    # Scale slightly to ensure no floating point overlap issues
    SCALE = 1.0 + 1e-9
    pts_scaled = []
    for x, y in best_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    
    pts_scaled = np.array(pts_scaled)
    
    outer_normals = np.array([
        [math.cos(best_beta + math.pi / 6 + k * math.pi / 3),
         math.sin(best_beta + math.pi / 6 + k * math.pi / 3)]
        for k in range(6)
    ])
    centered = pts_scaled - np.array([cx, cy])
    projs = centered @ outer_normals.T
    D_scaled = np.max(np.abs(projs))
    S_scaled = (D_scaled + APOTHEM) / APOTHEM

    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
