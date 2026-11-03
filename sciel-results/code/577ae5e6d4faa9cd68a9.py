import math
import numpy as np
import time
from scipy.optimize import linprog, minimize_scalar
import random

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Axial to Cartesian
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    # Check if two hexagons at positions p1, p2 (side 1) overlap
    # Distance between centers must be > 2 * apothem = sqrt(3)
    # Actually, for regular hexagons of side 1, the minimum distance between centers
    # to be disjoint is 2 * apothem = sqrt(3) if they are aligned?
    # No, the minimum distance is 2 * apothem = sqrt(3) only if they touch at edges.
    # If they touch at vertices, distance is 2.
    # The condition for non-overlap is distance > 2 * apothem = sqrt(3) is NOT sufficient.
    # Two hexagons overlap if distance < 2 * apothem = sqrt(3) is false?
    # Let's use the property that the union of two hexagons is convex if they are close.
    # A simpler check: The distance between centers must be at least 2 * apothem = sqrt(3)
    # is the condition for them to NOT touch?
    # Actually, if distance < 2 * apothem, they definitely overlap.
    # If distance is between 2 * apothem and 2, they might touch at vertices or be disjoint.
    # For safety, we require distance >= 2 * apothem + epsilon.
    # 2 * apothem = sqrt(3) ≈ 1.732.
    
    def is_valid_config(pts):
        n = len(pts)
        min_dist_sq = 3.0 + 1e-6 # (sqrt(3) + eps)^2 approx
        for i in range(n):
            for j in range(i+1, n):
                dx = pts[i, 0] - pts[j, 0]
                dy = pts[i, 1] - pts[j, 1]
                if dx*dx + dy*dy < min_dist_sq:
                    return False
        return True

    # Generate base lattice configurations
    candidates = []
    
    # Original candidates from seed
    axial_1 = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (1, 1), (-1, 2), (-2, 1), (2, -1)]
    axial_2 = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 0), (-2, 0), (0, 2), (0, -2)]
    axial_3 = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (-1, 1), (3, 1)]
    axial_4 = [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2)]
    axial_5 = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3)]
    axial_6 = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3)]
    axial_7 = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (2, 0), (-2, 0), (0, 2), (0, -2)]
    axial_8 = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]
    axial_9 = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 1), (-2, 2), (-1, 3), (1, -2)]
    axial_10 = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (1, 3), (2, 3)]
    
    candidates.extend([axial_1, axial_2, axial_3, axial_4, axial_5, axial_6, axial_7, axial_8, axial_9, axial_10])

    # Generate more random connected subsets
    # We define a grid of axial coordinates
    grid = []
    for q in range(-4, 5):
        for r in range(-4, 5):
            grid.append((q, r))
            
    # Map axial to index for quick lookup
    axial_to_idx = {a: i for i, a in enumerate(grid)}
    
    def get_neighbors(q, r):
        return [(q+1, r), (q-1, r), (q, r+1), (q, r-1), (q+1, r-1), (q-1, r+1)]

    def random_connected_subset(start, size):
        subset = {start}
        current = start
        while len(subset) < size:
            neighbors = get_neighbors(current[0], current[1])
            # Filter neighbors that are in grid and not in subset
            valid_neighbors = [n for n in neighbors if n in axial_to_idx and n not in subset]
            if not valid_neighbors:
                # Try to expand from any existing node
                added = False
                for node in list(subset):
                    nbs = get_neighbors(node[0], node[1])
                    vn = [n for n in nbs if n in axial_to_idx and n not in subset]
                    if vn:
                        subset.add(random.choice(vn))
                        added = True
                        break
                if not added:
                    return None
            else:
                subset.add(random.choice(valid_neighbors))
            current = random.choice(list(subset))
        return list(subset)

    # Generate 100 random configurations
    random.seed(42)
    for _ in range(100):
        start = random.choice(grid)
        config = random_connected_subset(start, N)
        if config:
            candidates.append(config)

    # Deduplicate configurations (by sorting tuples)
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    candidates = unique_candidates

    def calc_S_for_beta(pts, beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        
        n_pts = len(pts)
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for i in range(n_pts):
                px, py = pts[i]
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

    def exact_S(pts, beta, cx, cy):
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        pts_centered = pts - np.array([cx, cy])
        projs = np.abs(pts_centered @ normals.T)
        max_proj = float(np.max(projs))
        S = (max_proj + APOTHEM) / APOTHEM
        return S

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None
        
        # Coarse grid search over beta
        n_grid = 20
        for i in range(n_grid + 1):
            beta = i * (math.pi / 6) / n_grid
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Refine beta
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                refine_starts = [best_beta, 0.0, math.pi/12]
                for rb in refine_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    lo = max(0, rb - 0.2)
                    hi = min(math.pi/6, rb + 0.2)
                    res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], bounds=(lo, hi), method='bounded',
                                         options={'maxiter': 100, 'xatol': 1e-10})
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

    # Process candidates
    for idx, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        pts = axial_to_cartesian(axial)
        
        # Check validity
        if not is_valid_config(pts):
            continue
            
        # Try original
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center
            
        # Try jittered versions
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN - 10:
            continue
            
        for _ in range(5):
            jittered_pts = pts + np.random.normal(0, 0.05, size=pts.shape)
            # Re-center jittered points to keep them near origin
            jittered_pts -= np.mean(jittered_pts, axis=0)
            
            if not is_valid_config(jittered_pts):
                continue
                
            S, beta, center = optimize_shape(jittered_pts)
            if S < best_global_S:
                best_global_S = S
                best_global_pts = jittered_pts
                best_global_beta = beta
                best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        # Fallback to first valid candidate
        for axial in candidates:
            pts = axial_to_cartesian(axial)
            if is_valid_config(pts):
                best_global_pts = pts
                best_global_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
                best_global_beta = 0.0
                max_proj = 0
                for k in range(6):
                    ang = math.pi / 6.0 + k * math.pi / 3.0
                    nx, ny = math.cos(ang), math.sin(ang)
                    projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
                    max_proj = max(max_proj, np.max(projs), -np.min(projs))
                best_global_S = (max_proj + APOTHEM) / APOTHEM
                break

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Final safety scale
    SCALE = 1.0 + 1e-12
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    
    pts_scaled = np.array(pts_scaled)
    
    max_proj = 0.0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
    
    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM
    
    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
