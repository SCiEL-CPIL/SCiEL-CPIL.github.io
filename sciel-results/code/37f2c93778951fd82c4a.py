import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar
from itertools import combinations

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def _time_left():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)

def construct():
    if _time_left() <= 0:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate the 19-point hexagonal lattice (radius 2)
    # Axial coordinates (q, r)
    lattice_points = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                lattice_points.append((q, r))
    
    # Identify the 7-point core (radius 1)
    core_points = [(q, r) for q, r in lattice_points if abs(q) <= 1 and abs(r) <= 1 and abs(q+r) <= 1]
    # Identify the 12-point ring 2
    ring2_points = [(q, r) for q, r in lattice_points if (q, r) not in core_points]
    
    # Generate candidates: Core + 4 points from Ring 2
    candidates = []
    core_set = set(core_points)
    
    # We also add some other potential structures just in case
    # 1. The 7-core + 4 from Ring 2
    for combo in combinations(ring2_points, 4):
        pts = core_points + list(combo)
        candidates.append(pts)
        
    # 2. Other structures: e.g., 6 core + 5 ring 2?
    # To save time, let's stick to the 495 candidates first.
    # If time permits, we can add more.
    
    # Convert axial to cartesian
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def calc_S_for_beta_fast(pts, beta):
        """
        Fast calculation of S for a given beta.
        Finds optimal center via LP or simple logic.
        """
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

    def optimize_shape_fast(pts):
        """
        Optimizes center and rotation for a given set of points.
        Uses grid search + refinement.
        """
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        # Grid search for beta
        # The function is periodic with pi/6.
        # 60 steps should be enough for screening.
        for i in range(60):
            if _time_left() <= 0:
                break
            beta = i * (math.pi / 6) / 60
            S, center, D = calc_S_for_beta_fast(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Refine best beta
        if best_center is not None:
            try:
                res = minimize_scalar(lambda b: calc_S_for_beta_fast(pts, b)[0],
                                      bounds=(best_beta - (math.pi/6)/60, best_beta + (math.pi/6)/60),
                                      method='bounded', options={'maxiter': 100, 'xatol': 1e-10})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta_fast(pts, best_beta)
            except Exception:
                pass
        
        return best_S, best_beta, best_center

    def optimize_shape_high_precision(pts, init_beta, init_center):
        """
        High precision optimization using NM and LP.
        """
        best_S = float('inf')
        best_beta = init_beta
        best_center = init_center
        
        # Exact NM
        def exact_S(params):
            cx, cy, beta = params
            beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
            normals = np.array([
                (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
                 math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
                for k in range(6)
            ])
            pts_centered = pts - np.array([cx, cy])
            projs = pts_centered @ normals.T
            D = np.max(np.abs(projs))
            return (D + APOTHEM) / APOTHEM

        try:
            initial = [init_center[0], init_center[1], init_beta]
            res = minimize(exact_S, initial, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-14})
            if res.fun < best_S:
                best_S = res.fun
                best_center = (res.x[0], res.x[1])
                best_beta = ((res.x[2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
        except Exception:
            pass
        
        # LP Polish
        try:
            if _time_left() > 0:
                cx, cy = best_center
                for db in [-0.001, 0.001, 0.0]:
                    if _time_left() <= 0:
                        break
                    beta = ((best_beta + db) % (math.pi / 6) + math.pi / 6) % (math.pi / 6)
                    S, center, D = calc_S_for_beta_fast(pts, beta)
                    if S < best_S:
                        best_S = S
                        best_beta = beta
                        best_center = center
        except Exception:
            pass
        
        return best_S, best_beta, best_center

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None
    
    # Evaluate all 495 candidates
    for i, axial in enumerate(candidates):
        if _time_left() <= 0:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape_fast(pts)
        
        # Keep top 10 for high precision
        # We can just run high precision on the best found so far
        # Or store top 10 and process at the end.
        # Given 495 is small, we can run high precision on the best few.
        
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center
            
    # Run high precision on the best candidate found
    if best_global_pts is not None:
        S, beta, center = optimize_shape_high_precision(best_global_pts, best_global_beta, best_global_center)
        best_global_S = S
        best_global_beta = beta
        best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        # Fallback to first candidate
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])),
                              float(np.mean(best_global_pts[:, 1])))
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

    SCALE = 1.0 + 1e-10
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
