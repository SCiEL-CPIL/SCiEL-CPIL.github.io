import math
import numpy as np
import time
from scipy.optimize import minimize_scalar, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
MIN_DIST = math.sqrt(3)
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Initial configuration: Hexagonal cluster
    # Center at (0,0)
    # Ring 0: (0,0)
    # Ring 1: 6 neighbors
    # Ring 2: 5 more to make 11
    # Axial coords for a compact shape:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (1,1), (-1,2), (-2,1), (2,-1)
    # This is the same as axial_1 in the previous code, which is good.
    
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append([x, y])
        return np.array(pts)

    initial_axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    pts = axial_to_cartesian(initial_axial)

    def calc_S_for_pts(pts, beta):
        # Center points
        cx0 = np.mean(pts[:, 0])
        cy0 = np.mean(pts[:, 1])
        pts_centered = pts - np.array([cx0, cy0])

        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))

        max_proj = 0.0
        for nx, ny in normals:
            projs = pts_centered[:, 0] * nx + pts_centered[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))

        D = max_proj
        S = (D + APOTHEM) / APOTHEM
        return S, (cx0, cy0)

    def optimize_beta(pts):
        # Minimize S over beta
        def obj(beta):
            return calc_S_for_pts(pts, beta)[0]
        
        # Grid search first
        best_S = float('inf')
        best_beta = 0.0
        for i in range(100):
            beta = i * (math.pi / 6) / 99.0
            S, _ = calc_S_for_pts(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                
        # Refine
        try:
            res = minimize_scalar(obj, bounds=(0, math.pi / 6), method='bounded', options={'maxiter': 100})
            if res.fun < best_S:
                best_S = res.fun
                best_beta = res.x
        except Exception:
            pass
            
        return best_S, best_beta

    def is_valid(pts):
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                if np.linalg.norm(pts[i] - pts[j]) < MIN_DIST - 1e-9:
                    return False
        return True

    # Local Search
    current_pts = pts.copy()
    current_S, current_beta = optimize_beta(current_pts)
    
    best_pts = current_pts.copy()
    best_S = current_S
    best_beta = current_beta
    
    start_time = time.time()
    temp = 1.0
    cooling = 0.99
    
    while time.time() - start_time < TIME_LIMIT - SAFETY_MARGIN - 10.0:
        # Perturb
        new_pts = current_pts.copy()
        idx = np.random.randint(0, N)
        # Random direction and small step
        angle = np.random.uniform(0, 2 * math.pi)
        step = np.random.uniform(0.01, 0.2)
        new_pts[idx][0] += step * math.cos(angle)
        new_pts[idx][1] += step * math.sin(angle)
        
        if not is_valid(new_pts):
            continue
            
        new_S, new_beta = optimize_beta(new_pts)
        
        if new_S < current_S:
            current_pts = new_pts
            current_S = new_S
            current_beta = new_beta
            if new_S < best_S:
                best_S = new_S
                best_beta = new_beta
                best_pts = new_pts.copy()
        else:
            # Accept with probability
            delta = new_S - current_S
            if np.random.rand() < math.exp(-delta / temp):
                current_pts = new_pts
                current_S = new_S
                current_beta = new_beta
                
        temp *= cooling

    # Final optimization of best_pts
    best_S, best_beta = optimize_beta(best_pts)
    _, center = calc_S_for_pts(best_pts, best_beta)
    
    # Prepare output
    # Scale points slightly to ensure no overlap due to floating point
    SCALE = 1.0 + 1e-12
    cx, cy = center
    pts_scaled = []
    for x, y in best_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    pts_scaled = np.array(pts_scaled)
    
    # Recalculate S for scaled points
    S_scaled, _ = calc_S_for_pts(pts_scaled, best_beta)
    
    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(0.0)]) # Angle 0 for inner hexagons
    
    center_out = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(math.degrees(best_beta))
    
    return inner, center_out, side, rotation
