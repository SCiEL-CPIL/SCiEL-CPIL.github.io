import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # --- Generate Candidates ---
    # We generate all sets of 11 axial coordinates that form a connected cluster
    # and have a small bounding box to ensure compactness.
    
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def get_neighbors(q, r):
        return [(q+1, r), (q-1, r), (q, r+1), (q, r-1), (q+1, r-1), (q-1, r+1)]

    def generate_clusters():
        clusters = set()
        # Start from (0,0) and do a DFS/BFS to add 10 more hexagons
        # We restrict the search space to avoid huge clusters
        # Max coordinate range: -3 to 3
        
        def dfs(current_cluster, count):
            if count == N:
                # Canonicalize: sort the tuple of tuples
                key = tuple(sorted(current_cluster))
                clusters.add(key)
                return
            
            # Find all possible neighbors to add
            added = set()
            for q, r in current_cluster:
                for nq, nr in get_neighbors(q, r):
                    if (nq, nr) not in current_cluster:
                        # Check bounds to keep cluster compact
                        if -4 <= nq <= 4 and -4 <= nr <= 4 and -4 <= nq+nr <= 4:
                            if (nq, nr) not in added:
                                added.add((nq, nr))
                                new_cluster = list(current_cluster)
                                new_cluster.append((nq, nr))
                                dfs(new_cluster, count + 1)
            
        start = [(0,0)]
        dfs(start, 1)
        
        # Convert sets to lists for consistency
        return [list(c) for c in clusters]

    candidates = generate_clusters()
    
    # Deduplicate by shape (rotation/reflection invariant) is complex, 
    # but we can just evaluate all. The number of clusters is manageable.
    # To speed up, we can prune candidates that are clearly worse based on bounding box area.
    
    def bbox_area(axial_list):
        qs = [p[0] for p in axial_list]
        rs = [p[1] for p in axial_list]
        # Approximate area in cartesian
        pts = axial_to_cartesian(axial_list)
        if len(pts) < 3: return 0
        # Simple shoelace or just max extent
        return (np.max(pts[:,0]) - np.min(pts[:,0])) * (np.max(pts[:,1]) - np.min(pts[:,1]))

    # Sort by estimated area to find good ones early
    candidates.sort(key=bbox_area)
    
    # Limit candidates to avoid timeout
    MAX_CANDIDATES = 1000
    candidates = candidates[:MAX_CANDIDATES]

    def calc_S_for_beta(pts, beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for px, py in pts:
                proj = px * nx + py * ny
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-proj)
                A_ub.append([nx, ny, -1.0])
                b_ub.append(proj)
        
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

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None
        
        # 1. Check high-symmetry orientations first (0, 30, 60 deg)
        # These are often optimal for lattice packings
        priority_betas = [0.0, math.pi/6, math.pi/3, math.pi/12, math.pi/4, math.pi/5]
        
        for beta in priority_betas:
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # 2. Coarse grid search
        # Use a coarser grid to save time, then refine
        num_steps = 36
        for i in range(num_steps):
            beta = i * (math.pi / 6) / (num_steps - 1)
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # 3. Refine around best
        # Use minimize_scalar on the exact objective
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                lo = max(0, best_beta - math.pi/36)
                hi = min(math.pi/6, best_beta + math.pi/36)
                
                # If best_beta is near boundary, expand range
                if best_beta < math.pi/72:
                    lo = 0.0
                if best_beta > math.pi/6 - math.pi/72:
                    hi = math.pi/6
                    
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], bounds=(lo, hi), method='bounded',
                                     options={'maxiter': 100, 'xatol': 1e-8})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
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
            
        # Quick rejection: if bounding box is too large, skip?
        # The LP will handle it, but we can skip if area is huge.
        
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center
            
            # Early exit if we find a very good solution?
            # Theoretical minimum for N=11 is around S=3 or 4?
            # If we get S < 3.5, it's likely near optimal.
            if best_global_S < 3.2:
                # We can break or continue with fewer candidates
                pass

    if best_global_center is None or best_global_S == float('inf'):
        # Fallback
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
    
    # Scale points slightly to ensure they fit inside the hexagon defined by S
    # The LP finds the minimal S for the points. 
    # To be safe, we can return the points as is, and S as the minimal enclosing size.
    # The evaluator checks if points are inside the hexagon of side S.
    # Since S is the *minimal* enclosing size, the points are exactly on the boundary or inside.
    # Due to floating point errors, they might be slightly outside.
    # We can scale the points down slightly relative to the center.
    
    SCALE = 1.0 - 1e-9
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    
    pts_scaled = np.array(pts_scaled)
    
    # Recalculate S to be safe (should be slightly smaller or same)
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
