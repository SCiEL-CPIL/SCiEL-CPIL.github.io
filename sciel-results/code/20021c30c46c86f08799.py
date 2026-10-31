import math
import numpy as np
import time
from scipy.optimize import differential_evolution

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 15.0 # Increased safety margin

def construct():
    # Check if we are already close to the time limit
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Return a trivial valid solution: 1 hexagon
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate 11 points on a triangular lattice
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    # Scale factor: Start slightly expanded to be safe
    SCALE = 1.001
    
    pts = np.array([
        (math.sqrt(3) * (q + r / 2.0) * SCALE, 1.5 * r * SCALE)
        for q, r in axial
    ])
    
    # Initial best solution: the lattice one
    def calc_S_for_config(pts, beta=0.0):
        centroid = np.mean(pts, axis=0)
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        max_S = 0.0
        for nx, ny in normals:
            rel_pts = pts - centroid
            projections = rel_pts[:, 0] * nx + rel_pts[:, 1] * ny
            abs_proj = np.max(np.abs(projections))
            required_S = (abs_proj + APOTHEM) / APOTHEM
            if required_S > max_S:
                max_S = required_S
        return max_S

    def is_valid(pts, tol=1e-6):
        min_dist = float('inf')
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_dist:
                    min_dist = d
        return min_dist >= math.sqrt(3) - tol

    best_S = calc_S_for_config(pts)
    best_pts = pts.copy()
    
    if not is_valid(best_pts):
        # If even the initial is invalid (shouldn't happen with SCALE=1.001), increase scale
        SCALE = 1.01
        pts = np.array([
            (math.sqrt(3) * (q + r / 2.0) * SCALE, 1.5 * r * SCALE)
            for q, r in axial
        ])
        best_pts = pts.copy()
        best_S = calc_S_for_config(pts)

    # Define objective function for optimization
    # x: [x1, y1, ..., x11, y11]
    def objective(x):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf')
        
        pts_opt = x.reshape(N, 2)
        
        # Calculate S
        S = calc_S_for_config(pts_opt)
        
        # Calculate overlap penalty
        min_dist = float('inf')
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pts_opt[i] - pts_opt[j])
                if d < min_dist:
                    min_dist = d
        
        if min_dist < math.sqrt(3):
            penalty = 1000.0 * (math.sqrt(3) - min_dist) ** 2
        else:
            penalty = 0.0
            
        return S + penalty

    # Define bounds for Differential Evolution
    # The hexagons should stay within a reasonable range
    # Max distance from center is roughly 2 * side * sqrt(3)
    bound = 5.0
    bounds = [(-bound, bound)] * (2 * N)

    try:
        # Use Differential Evolution
        # maxiter and popsize tuned for time limit
        # N=11, 22 variables.
        res = differential_evolution(objective, bounds, maxiter=200, popsize=15, 
                                      tol=1e-4, seed=42, polish=True, 
                                      init='sobol')
        
        if res.fun < best_S and res.fun < float('inf'):
            pts_res = res.x.reshape(N, 2)
            
            # Strict validity check
            if is_valid(pts_res, tol=1e-5):
                S_res = calc_S_for_config(pts_res)
                if S_res < best_S:
                    best_S = S_res
                    best_pts = pts_res
                    
    except Exception:
        pass

    # Prepare output
    inner = []
    # All inner hexagons have rotation 0
    angle_deg = 0.0
    for x, y in best_pts:
        inner.append([float(x), float(y), angle_deg])
        
    centroid = np.mean(best_pts, axis=0)
    center = [float(centroid[0]), float(centroid[1])]
    side = float(best_S)
    rotation = angle_deg

    return inner, center, side, rotation
