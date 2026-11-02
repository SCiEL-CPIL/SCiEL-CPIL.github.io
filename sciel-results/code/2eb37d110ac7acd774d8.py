import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0  # Increased safety margin

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # The minimum distance between centers of two disjoint unit hexagons is sqrt(3)
    MIN_DIST = math.sqrt(3)
    
    # Initial guess: A central hexagon and 6 around it, plus 4 more.
    # Standard hexagonal packing coordinates for 11 hexagons.
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    # A reasonable starting configuration for 11 hexagons
    start_axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (0, 2), (-2, 2), (2, -2)
    ]
    
    start_pts = axial_to_cartesian(start_axial)
    
    def get_outer_side(pts, cx, cy, beta):
        # pts: (N, 2)
        # Returns the side length of the minimal hexagon of angle beta enclosing pts, centered at (cx, cy)
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        centered = pts - np.array([cx, cy])
        projs = centered @ normals.T # (N, 6)
        max_proj = np.max(np.abs(projs))
        D = max_proj
        S = (D + APOTHEM) / APOTHEM
        return S

    def check_validity(pts, tol=1e-8):
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(pts[i] - pts[j])
                if dist < MIN_DIST - tol:
                    return False
        return True

    def objective(params):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf')
        
        n_vars = len(params)
        pts = params[:2*N].reshape(N, 2)
        cx = params[2*N]
        cy = params[2*N+1]
        beta = params[2*N+2]
        
        S = get_outer_side(pts, cx, cy, beta)
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(pts[i] - pts[j])
                if dist < MIN_DIST:
                    penalty += (MIN_DIST - dist) ** 2
        
        return S + 1000.0 * penalty  # Increased penalty weight

    x0 = np.zeros(2*N + 3)
    x0[:2*N] = start_pts.flatten()
    x0[2*N] = 0.0
    x0[2*N+1] = 0.0
    x0[2*N+2] = 0.0

    best_S = float('inf')
    best_params = x0.copy()
    best_pts = start_pts.copy()
    best_cx = 0.0
    best_cy = 0.0
    best_beta = 0.0
    best_valid = check_validity(best_pts)

    try:
        # Use Nelder-Mead
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        
        if res.fun < best_S:
            best_S = res.fun
            best_params = res.x
            
            # Check validity of the result
            cand_pts = best_params[:2*N].reshape(N, 2)
            if check_validity(cand_pts):
                best_pts = cand_pts
                best_cx = best_params[2*N]
                best_cy = best_params[2*N+1]
                best_beta = best_params[2*N+2]
                best_valid = True
            # If invalid, we keep the previous best valid state (which is start_pts initially)
            
        # Try a few random perturbations
        for _ in range(5):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            perturb = np.random.normal(0, 0.3, 2*N)
            x_pert = x0.copy()
            x_pert[:2*N] += perturb
            
            res_p = minimize(objective, x_pert, method='Nelder-Mead', 
                             options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            if res_p.fun < best_S:
                best_S = res_p.fun
                cand_pts = res_p.x[:2*N].reshape(N, 2)
                if check_validity(cand_pts):
                    best_pts = cand_pts
                    best_cx = res_p.x[2*N]
                    best_cy = res_p.x[2*N+1]
                    best_beta = res_p.x[2*N+2]
                    best_valid = True

    except Exception as e:
        pass

    # Ensure we have a valid configuration
    if not best_valid:
        # Fall back to start_pts which is known to be valid
        best_pts = start_pts
        best_cx = 0.0
        best_cy = 0.0
        best_beta = 0.0

    # Calculate final S
    final_S = get_outer_side(best_pts, best_cx, best_cy, best_beta)
    
    # Construct output
    inner = []
    angle_deg = math.degrees(best_beta)
    for x, y in best_pts:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(best_cx), float(best_cy)]
    side = float(final_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
