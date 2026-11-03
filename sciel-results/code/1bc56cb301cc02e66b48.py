import math
import numpy as np
import time
from scipy.optimize import minimize_scalar, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Function to calculate minimal enclosing hexagon side length for a set of points
    # assuming inner hexagons are aligned with the outer hexagon
    def calc_S_for_beta(pts, beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None

        # Center the points at origin for symmetry
        cx0 = np.mean(pts[:, 0])
        cy0 = np.mean(pts[:, 1])
        pts_centered = pts - np.array([cx0, cy0])

        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))

        # Compute max projection in each normal direction
        max_proj = 0.0
        for nx, ny in normals:
            projs = pts_centered[:, 0] * nx + pts_centered[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))

        D = max_proj
        S = (D + APOTHEM) / APOTHEM
        return S, (cx0, cy0), D

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        # Coarse grid search over beta
        for i in range(73):
            beta = i * (math.pi / 6) / 72
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center

        # Refine beta with bounded scalar optimization
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 500, 'xatol': 1e-14})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        return best_S, best_beta, best_center

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    # Generate candidates using a greedy growth algorithm
    # and some predefined good shapes
    candidates = []
    
    # 1. Predefined good shapes (from previous logic, but selected for compactness)
    # A 3x3 hexagon is 9 hexagons. Add 2 more.
    # Hexagon of radius 1 (7 hexagons) + 4 more?
    
    # Let's generate candidates by growing a cluster
    def generate_candidate(strategy):
        # strategy: 0 = greedy min-S, 1 = greedy min-area, 2 = random
        pts_axial = [(0, 0)]
        for i in range(N - 1):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            best_pos = None
            best_score = float('inf')
            
            # Candidate positions: neighbors of existing points, and some further out
            candidates_pos = set()
            for q, r in pts_axial:
                for dq, dr in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
                    candidates_pos.add((q + dq, r + dr))
                # Add some "second neighbor" positions to allow gaps or wider shapes
                for dq, dr in [(2, 0), (-2, 0), (0, 2), (0, -2), (2, -2), (-2, 2)]:
                    candidates_pos.add((q + dq, r + dr))
            
            # Remove existing positions
            candidates_pos -= set(pts_axial)
            
            if strategy == 2:
                # Random choice
                if candidates_pos:
                    pos = list(candidates_pos)[np.random.randint(len(candidates_pos))]
                    pts_axial.append(pos)
                    continue
            
            for pos in candidates_pos:
                new_pts = pts_axial + [pos]
                # Quick score: just use beta=0 for speed in greedy step
                pts_cart = axial_to_cartesian(new_pts)
                S, _, _ = calc_S_for_beta(pts_cart, 0.0)
                if S < best_score:
                    best_score = S
                    best_pos = pos
            
            if best_pos is not None:
                pts_axial.append(best_pos)
            else:
                # Fallback if no candidates
                pts_axial.append((len(pts_axial), 0))
        
        return pts_axial

    # Run several greedy generations with different random seeds/strategies
    np.random.seed(42)
    for _ in range(5):
        candidates.append(generate_candidate(0))
    
    # Add some specific known good configurations
    # 1. 3x3 block + 2 on the side
    base_3x3 = []
    for q in range(-1, 2):
        for r in range(-1, 2):
            if abs(q) + abs(r) + abs(q+r) <= 2: # Hexagon of radius 1? No, this is radius 1 (7 hexes)
                base_3x3.append((q, r))
    # Actually, a hexagon of "side 2" (radius 1) has 7 hexagons.
    # A hexagon of "side 3" (radius 2) has 19 hexagons.
    # We want 11.
    
    # Let's manually define a few compact ones
    # 1. Center + 6 neighbors + 4 more
    c1 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (-2,0), (0,2), (0,-2)]
    candidates.append(c1)
    
    # 2. 2x5 strip
    c2 = [(q, r) for q in range(5) for r in range(2)]
    candidates.append(c2)
    
    # 3. 3x4 block
    c3 = [(q, r) for q in range(3) for r in range(4)]
    candidates.append(c3)
    
    # 4. Triangle-ish
    c4 = [(q, r) for q in range(4) for r in range(3) if q + r < 4] # Roughly
    # Ensure 11 points
    while len(c4) < 11:
        c4.append((len(c4), 0))
    candidates.append(c4)

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

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

    # Use a very small safety margin to ensure no overlap
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
