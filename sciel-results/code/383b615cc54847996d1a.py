import math
import numpy as np
import time

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Precompute axial to cartesian
    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return (x, y)

    def get_points(axial_coords):
        return np.array([axial_to_cartesian(q, r) for q, r in axial_coords])

    def enclosing_hexagon_side(pts, beta):
        """
        Compute the side length S of the smallest enclosing regular hexagon
        with rotation angle beta (relative to pts).
        pts: (n, 2) array
        beta: rotation angle of the hexagon
        Returns: (S, cx, cy)
        """
        if len(pts) == 0:
            return 0.0, 0.0, 0.0
        
        normals = np.array([
            [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
            for k in range(6)
        ])
        
        # Projection of pts onto each normal
        # proj[i, j] = pts[i] @ normals[j]
        proj = pts @ normals.T
        
        # For each normal direction k, we have:
        # max_i (pts[i] @ n_k) - min_i (pts[i] @ n_k) = width_k
        # But the hexagon is centered at (cx, cy), so:
        # (pts[i] - center) @ n_k <= apothem
        # pts[i] @ n_k - center @ n_k <= apothem
        # pts[i] @ n_k <= center @ n_k + apothem
        # Similarly: -pts[i] @ n_k <= -center @ n_k + apothem
        #            pts[i] @ n_k >= center @ n_k - apothem
        #
        # So for each k:
        # max_i pts[i] @ n_k <= center @ n_k + apothem
        # min_i pts[i] @ n_k >= center @ n_k - apothem
        # => center @ n_k >= max_i pts[i] @ n_k - apothem
        # => center @ n_k <= min_i pts[i] @ n_k + apothem
        #
        # Let h_k = center @ n_k
        # h_k in [max_proj_k - apothem, min_proj_k + apothem]
        #
        # We want to minimize S = (D + apothem) / apothem where D is the "radius"
        # Actually, the hexagon side length S relates to apothem by apothem = S * sqrt(3)/2
        # So S = 2 * apothem / sqrt(3) = 2 * apothem / sqrt(3)
        # Wait, apothem = S * sqrt(3) / 2, so S = 2 * apothem / sqrt(3)
        #
        # The constraint is that for all points and all normals:
        # | (pts[i] - center) @ n_k | <= apothem
        #
        # Let D = max over k of max_i | (pts[i] - center) @ n_k |
        # Then we need apothem >= D, so S >= 2D / sqrt(3)
        #
        # We want to minimize D over center.
        # D = max_k max_i | (pts[i] - center) @ n_k |
        #
        # This is a convex optimization. Let's use LP.
        # Variables: cx, cy, D
        # Minimize D
        # Subject to:
        # For each k, i: (pts[i] - center) @ n_k <= D
        #                -(pts[i] - center) @ n_k <= D
        #
        # => pts[i] @ n_k - center @ n_k <= D
        #    center @ n_k - pts[i] @ n_k <= D
        
        from scipy.optimize import linprog
        
        c = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        for k in range(6):
            nx, ny = normals[k]
            for i in range(len(pts)):
                px, py = pts[i]
                # (px - cx) * nx + (py - cy) * ny <= D
                # -cx * nx - cy * ny + px * nx + py * ny <= D
                # -nx * cx - ny * cy - D <= - (px * nx + py * ny)
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                
                # -(px - cx) * nx - (py - cy) * ny <= D
                # cx * nx + cy * ny - px * nx - py * ny <= D
                # nx * cx + ny * cy - D <= px * nx + py * ny
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                cx, cy, D = res.x
                S = 2.0 * D / math.sqrt(3)
                return S, cx, cy
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def enclosing_hexagon_side_fast(pts, beta):
        """Fast version without LP - just compute D directly assuming center is centroid."""
        if len(pts) == 0:
            return 0.0, 0.0, 0.0
        
        normals = np.array([
            [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
            for k in range(6)
        ])
        
        center = np.mean(pts, axis=0)
        shifted = pts - center
        proj = shifted @ normals.T
        D = np.max(np.abs(proj))
        S = 2.0 * D / math.sqrt(3)
        return S, center[0], center[1]

    # Generate candidate axial configurations
    candidates = []
    
    # Hexagonal spiral: center + ring of 6 + partial ring of 4
    spiral1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, -2)
    ]
    candidates.append(spiral1)
    
    # Alternative spiral
    spiral2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (2, 1), (-2, 0), (0, -2)
    ]
    candidates.append(spiral2)
    
    # Another arrangement
    spiral3 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    candidates.append(spiral3)
    
    # Linear-ish arrangement
    line = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (3, 0), (4, 0), (5, 0)
    ]
    candidates.append(line)
    
    # Compact 4x3 grid-like
    grid = []
    for r in range(-1, 2):
        for q in range(-1, 4):
            if len(grid) < N:
                grid.append((q, r))
    candidates.append(grid)
    
    # Random clusters
    np.random.seed(42)
    for _ in range(5):
        random_axial = []
        for i in range(N):
            q = np.random.randint(-3, 4)
            r = np.random.randint(-3, 4)
            random_axial.append((int(q), int(r)))
        candidates.append(random_axial)
    
    best_S = float('inf')
    best_beta = 0.0
    best_center = None
    best_axial = None
    best_pts = None
    
    # For each candidate, try multiple rotation angles
    for axial_coords in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        pts = get_points(axial_coords)
        
        # Try a range of beta values
        for beta_idx in range(37):
            beta = beta_idx * (math.pi / 6.0) / 36.0
            
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            # Use fast estimate first
            S_fast, cx_fast, cy_fast = enclosing_hexagon_side_fast(pts, beta)
            
            # If fast estimate is promising, do the LP for exact solution
            if S_fast < best_S + 0.1:
                S, cx, cy = enclosing_hexagon_side(pts, beta)
                if S < best_S:
                    best_S = S
                    best_beta = beta
                    best_center = (cx, cy)
                    best_axial = axial_coords
                    best_pts = pts
        
        # Refine beta around best if found
        if best_S < float('inf'):
            # Try fine-tuning beta
            for delta_beta in [-0.01, -0.005, 0.005, 0.01]:
                beta_test = best_beta + delta_beta
                S_test, cx_test, cy_test = enclosing_hexagon_side(best_pts, beta_test)
                if S_test < best_S:
                    best_S = S_test
                    best_beta = beta_test
                    best_center = (cx_test, cy_test)
    
    # Fallback
    if best_center is None:
        best_axial = candidates[0]
        best_pts = get_points(best_axial)
        best_beta = 0.0
        best_S, best_center[0], best_center[1] = enclosing_hexagon_side(best_pts, best_beta)
    
    # Now construct output
    # The inner hexagons are at positions best_pts, each with angle = best_beta (in degrees)
    # The outer hexagon is centered at best_center with side best_S and rotation best_beta
    
    cx, cy = best_center
    angle_deg = math.degrees(best_beta)
    
    # Apply a tiny scale to ensure non-overlap safety
    SCALE = 1.0 + 1e-8
    pts_scaled = np.array([
        [cx + (x - cx) * SCALE, cy + (y - cy) * SCALE]
        for x, y in best_pts
    ])
    
    # Recompute S with scaled points
    S_final, cx_final, cy_final = enclosing_hexagon_side(pts_scaled, best_beta)
    
    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
    
    center = [float(cx_final), float(cy_final)]
    side = float(S_final)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
