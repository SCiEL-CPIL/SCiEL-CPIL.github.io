import math
import numpy as np
import time
from scipy.optimize import linprog, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

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

    def get_neighbors(q, r):
        return [
            (q+1, r), (q-1, r),
            (q, r+1), (q, r-1),
            (q+1, r-1), (q-1, r+1)
        ]

    def compactness_score(pts):
        # Minimize sum of squared distances from centroid
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        return np.sum((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)

    def grow_cluster(num_hex, seed=0):
        import random
        random.seed(seed)
        current = [(0, 0)]
        occupied = set(current)
        
        for _ in range(num_hex - 1):
            candidates = []
            for q, r in current:
                for nq, nr in get_neighbors(q, r):
                    if (nq, nr) not in occupied:
                        candidates.append((nq, nr))
            
            if not candidates:
                break
                
            # Score each candidate
            scored = []
            for q, r in candidates:
                new_set = current + [(q, r)]
                pts = axial_to_cartesian(new_set)
                score = compactness_score(pts)
                scored.append((score, q, r))
            
            scored.sort(key=lambda x: x[0])
            
            # Pick the best, or sometimes a random one from the top 3 to allow diversity
            if seed % 10 == 0 and len(scored) > 3:
                # Random pick from top 3
                idx = random.randint(0, 2)
                _, q, r = scored[idx]
            else:
                _, q, r = scored[0]
                
            current.append((q, r))
            occupied.add((q, r))
            
        return current

    # Generate candidates
    candidates = []
    # Add some known good ones
    known_good = [
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (-2,0), (0,2), (0,-2)],
        [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2), (1,3), (2,3)]
    ]
    candidates.extend(known_good)
    
    # Generate 200 random growth clusters
    for i in range(200):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        cand = grow_cluster(N, seed=i)
        if len(cand) == N:
            candidates.append(cand)

    def optimize_rigid(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        # Grid search for beta
        # Beta is the rotation of the outer hexagon relative to the lattice
        # Due to symmetry, we only need to search [0, pi/6]
        num_steps = 50
        for i in range(num_steps):
            beta = i * (math.pi / 6.0) / (num_steps - 1)
            
            normals = []
            for k in range(6):
                ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                normals.append((math.cos(ang), math.sin(ang)))
            
            c_obj = np.array([0.0, 0.0, 1.0])
            A_ub = []
            b_ub = []
            
            for nx, ny in normals:
                for px, py in pts:
                    # Constraint: | (p - c) . n | <= D
                    # (p - c) . n <= D  => -nx*cx - ny*cy - D <= -px*nx - py*ny
                    A_ub.append([-nx, -ny, -1.0])
                    b_ub.append(-(px * nx + py * ny))
                    
                    # -(p - c) . n <= D  => nx*cx + ny*cy - D <= px*nx + py*ny
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
                    if S < best_S:
                        best_S = S
                        best_beta = beta
                        best_center = (cx, cy)
            except Exception:
                pass

        return best_S, best_beta, best_center

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_rigid(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        # Fallback to a simple grid
        pts = axial_to_cartesian([(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (-2,0), (0,2), (0,-2)])
        best_global_center = (0.0, 0.0)
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Scale points slightly to ensure no floating point overlap issues
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
