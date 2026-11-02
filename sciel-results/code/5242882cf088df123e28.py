import math
import numpy as np
import time
from scipy.optimize import linprog, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 2.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate candidates
    candidates = set()
    
    # Base 7-hex cluster
    base = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    # Directions for expansion
    dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    # We want to add 4 hexes to the base cluster.
    # We will try all combinations of 4 "steps" from the perimeter.
    # To avoid duplicates and explosion, we will use a canonical form (sorting axial coords).
    
    # Heuristic: The optimal shape is likely close to a hexagon.
    # We will generate shapes by extending the 6 arms.
    
    # Represent a shape as a frozenset of axial coords.
    # Start with base.
    
    # We can use a simple BFS to find all connected sets of 11 hexes 
    # that contain the origin and have a small bounding box.
    
    # However, generating all is expensive.
    # Let's use the "arm extension" logic.
    # An arm is a sequence of hexes in a direction.
    
    # We will try all distributions of 4 extra hexes among the 6 arms.
    # This covers "star" shapes.
    
    # Also try "block" shapes.
    
    # Let's just generate a diverse set of candidates manually + programmatically.
    
    # 1. Star shapes (extensions of the 6 arms)
    # Distributions of 4 into 6 parts (non-negative integers)
    # Compositions of 4 into 6 parts: C(4+6-1, 6-1) = C(9,5) = 126.
    # For each distribution, we assign the counts to the 6 directions.
    # But the directions are symmetric, so we can fix the first non-zero count to be in dir 0?
    # No, just iterate all 126 compositions and apply to the 6 dirs.
    # Then rotate the resulting shape? No, the shape is defined by the counts.
    # We just need to generate the set of points.
    
    def gen_star_shape(counts):
        pts = set(base)
        for i, c in enumerate(counts):
            d = dirs[i]
            for k in range(1, c + 1):
                # Extend from the outermost point of the current arm
                # The arm in direction i currently has length 1 (from base).
                # So we add k=1..c to the existing arm.
                # The point at distance 1 is d[0]*1, d[1]*1.
                # The point at distance 2 is d[0]*2, d[1]*2.
                # So we add (d[0]*j, d[1]*j) for j in 1..c?
                # Wait, base already has (d[0]*1, d[1]*1).
                # So we add (d[0]*(1+j), d[1]*(1+j)) for j in 0..c-1.
                for j in range(c):
                    new_q = d[0] * (1 + j + 1) # 1 is base, +1 for next
                    new_r = d[1] * (1 + j + 1)
                    pts.add((new_q, new_r))
        return frozenset(pts)
    
    # Generate compositions of 4 into 6 parts
    def compositions(n, k):
        if k == 1:
            yield [n]
            return
        for i in range(n + 1):
            for rest in compositions(n - i, k - 1):
                yield [i] + rest

    for comp in compositions(4, 6):
        candidates.add(gen_star_shape(comp))
        
    # 2. Block shapes
    # 3x3 block is 9 hexes. Add 2.
    block_3x3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    # Add 2 hexes to the perimeter of 3x3
    # Perimeter points of 3x3:
    # (0,0) is corner.
    # We can add hexes adjacent to the perimeter.
    # Let's just add specific promising shapes.
    
    # Shape: 3x3 + 2 on a side
    shape_block_1 = frozenset([(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2), (-1,1), (-1,2)])
    candidates.add(shape_block_1)
    
    shape_block_2 = frozenset([(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2), (1,3), (2,3)])
    candidates.add(shape_block_2)
    
    # Shape: 2x5 block + 1
    shape_block_3 = frozenset([(0,0), (1,0), (2,0), (3,0), (4,0), (0,1), (1,1), (2,1), (3,1), (4,1), (0,2)])
    candidates.add(shape_block_3)
    
    # Shape: 1x11 line (worst case, but include for completeness)
    shape_line = frozenset([(i, 0) for i in range(11)])
    candidates.add(shape_line)
    
    # Convert candidates to Cartesian points
    def axial_to_cartesian(axial_set):
        pts = []
        for q, r in axial_set:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def calc_S_for_beta(pts, beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        
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

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None
        
        # Coarse grid search over beta
        # 12 steps for pi/6
        for i in range(13):
            beta = i * (math.pi / 6) / 12
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Refine beta
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 100, 'xatol': 1e-12})
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

    # Evaluate candidates
    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        # Fallback to a simple shape
        pts = axial_to_cartesian(list(candidates)[0])
        best_global_pts = pts
        best_global_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
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
    
    # Scale points to ensure non-overlap
    # The distance between centers in the grid is exactly sqrt(3).
    # We scale by 1 + epsilon to make it strictly greater.
    SCALE = 1.0 + 1e-9
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    
    pts_scaled = np.array(pts_scaled)
    
    # Recalculate S with scaled points
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
