import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Precompute inner hexagon normals (for overlap checking)
    inner_normals = np.array([
        [math.cos(math.pi/6 + k*math.pi/3), math.sin(math.pi/6 + k*math.pi/3)]
        for k in range(6)
    ])

    def compute_S_smooth(params, temperature=0.1):
        n_pts = N
        pts = params[:2*n_pts].reshape(n_pts, 2)
        cx = params[2*n_pts]
        cy = params[2*n_pts+1]
        beta = params[2*n_pts+2]
        
        # Compute outer normals
        outer_normals = np.array([
            [math.cos(beta + math.pi/6 + k*math.pi/3), math.sin(beta + math.pi/6 + k*math.pi/3)]
            for k in range(6)
        ])
        
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ outer_normals.T
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        
        max_val = np.max(all_projs)
        if max_val > 100:
            return 1e10
            
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        
        S = (D + APOTHEM) / APOTHEM
        
        # Overlap penalty
        penalty = 0.0
        for i in range(n_pts):
            for j in range(i+1, n_pts):
                d = pts[i] - pts[j]
                projs_inner = np.abs(d @ inner_normals.T)
                min_proj = np.min(projs_inner)
                if min_proj < math.sqrt(3):
                    violation = math.sqrt(3) - min_proj
                    penalty += violation * 50.0
                    
        return S + penalty

    def check_overlap(pts):
        n_pts = len(pts)
        for i in range(n_pts):
            for j in range(i+1, n_pts):
                d = pts[i] - pts[j]
                projs = np.abs(d @ inner_normals.T)
                if np.min(projs) < math.sqrt(3) - 1e-9:
                    return False
        return True

    def calc_S_exact(pts, cx, cy, beta):
        outer_normals = np.array([
            [math.cos(beta + math.pi/6 + k*math.pi/3), math.sin(beta + math.pi/6 + k*math.pi/3)]
            for k in range(6)
        ])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ outer_normals.T
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        D = np.max(all_projs)
        S = (D + APOTHEM) / APOTHEM
        return S, (cx, cy), D

    best_S = float('inf')
    best_params = None

    # Simulated Annealing
    # Initial guess: a dense grid-like configuration
    # Axial coordinates for a compact cluster
    axial = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    pts_init = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts_init.append((x, y))
    pts_init = np.array(pts_init)
    
    cx_init = float(np.mean(pts_init[:, 0]))
    cy_init = float(np.mean(pts_init[:, 1]))
    beta_init = 0.0

    current_params = np.concatenate([pts_init.ravel(), [cx_init, cy_init, beta_init]])
    current_S = compute_S_smooth(current_params, temperature=0.1)
    
    T = 0.5
    T_min = 0.001
    cooling = 0.995
    max_iter = 3000
    
    for i in range(max_iter):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        # Generate neighbor
        noise_scale = 0.1
        neighbor_params = current_params + np.random.normal(0, noise_scale, size=current_params.size)
        
        # Keep beta in reasonable range
        neighbor_params[-1] = ((neighbor_params[-1] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        
        neighbor_S = compute_S_smooth(neighbor_params, temperature=0.1)
        
        if neighbor_S < current_S:
            current_params = neighbor_params
            current_S = neighbor_S
        else:
            delta = neighbor_S - current_S
            if delta > 0:
                prob = math.exp(-delta / T)
            else:
                prob = 1.0
            if np.random.random() < prob:
                current_params = neighbor_params
                current_S = neighbor_S
                
        T *= cooling
        if T < T_min:
            break
            
        # Update best
        if current_S < best_S:
            best_S = current_S
            best_params = current_params.copy()

    # Polish with Nelder-Mead
    if best_params is not None and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        def polish_obj(params):
            return compute_S_smooth(params, temperature=0.01)
            
        try:
            res = minimize(polish_obj, best_params, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
        except Exception:
            pass

    # Extract best configuration
    n_pts = N
    best_pts = best_params[:2*n_pts].reshape(n_pts, 2)
    best_cx = best_params[2*n_pts]
    best_cy = best_params[2*n_pts+1]
    best_beta = best_params[2*n_pts+2]
    best_beta = ((best_beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
    
    # Verify overlap and recalculate exact S
    if check_overlap(best_pts):
        S_exact, center, D = calc_S_exact(best_pts, best_cx, best_cy, best_beta)
        if S_exact < best_S:
            best_S = S_exact
    else:
        # Fallback to a valid configuration if optimization failed
        best_S = (np.max(np.abs(pts_init)) + APOTHEM) / APOTHEM
        best_pts = pts_init
        best_cx = cx_init
        best_cy = cy_init
        best_beta = 0.0

    angle_deg = math.degrees(best_beta)
    
    inner = []
    for x, y in best_pts:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(best_cx), float(best_cy)]
    side = float(best_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
