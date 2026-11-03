import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar, linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def hexagon_vertices(cx, cy, side=1.0, angle=0.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    vertices = []
    for i in range(6):
        theta = angle + i * math.pi / 3.0
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        vertices.append((x, y))
    return np.array(vertices)

def hexagon_edges(cx, cy, side=1.0, angle=0.0):
    """Return edge normals and offsets for a hexagon (for LP constraints)."""
    # For a regular hexagon, the 6 edge normals are at angles angle + pi/6 + k*pi/3
    # The offset (apothem) is side * sqrt(3)/2
    apothem = side * math.sqrt(3) / 2.0
    normals = []
    offsets = []
    for i in range(6):
        theta = angle + math.pi / 6.0 + i * math.pi / 3.0
        nx = math.cos(theta)
        ny = math.sin(theta)
        normals.append((nx, ny))
        offsets.append(cx * nx + cy * ny - apothem)
    return np.array(normals), np.array(offsets)

def separation_constraint(h1_center, h1_angle, h2_center, h2_angle):
    """
    Compute the minimum translation vector to separate two overlapping hexagons.
    Returns (dx, dy, distance) where (dx, dy) is the direction to move h1 away from h2.
    Uses the fact that for convex polygons, the minimum separating axis is perpendicular to an edge.
    """
    v1 = hexagon_vertices(h1_center[0], h1_center[1], 1.0, h1_angle)
    v2 = hexagon_vertices(h2_center[0], h2_center[1], 1.0, h2_angle)
    
    # Check all edge normals from both hexagons
    min_dist = float('inf')
    best_dir = None
    
    for i in range(6):
        # Normal from h1
        theta1 = h1_angle + math.pi / 6.0 + i * math.pi / 3.0
        n1 = np.array([math.cos(theta1), math.sin(theta1)])
        
        # Project both hexagons onto this normal
        proj1 = v1 @ n1
        proj2 = v2 @ n1
        
        # For separation, we need max(proj1) <= min(proj2) or vice versa
        # The gap along this axis is: min(proj2) - max(proj1)
        gap = min(proj2) - max(proj1)
        
        if gap > min_dist:
            min_dist = gap
            best_dir = n1
    
    # Also check the opposite direction
    for i in range(6):
        theta2 = h2_angle + math.pi / 6.0 + i * math.pi / 3.0
        n2 = np.array([math.cos(theta2), math.sin(theta2)])
        
        proj1 = v1 @ n2
        proj2 = v2 @ n2
        
        gap = min(proj1) - max(proj2)
        
        if gap > min_dist:
            min_dist = gap
            best_dir = -n2
    
    if best_dir is None:
        # Hexagons are far apart, any direction works
        d = np.array(h1_center) - np.array(h2_center)
        norm = np.linalg.norm(d)
        if norm > 1e-10:
            best_dir = d / norm
        else:
            best_dir = np.array([1.0, 0.0])
        min_dist = norm
    
    return best_dir, min_dist

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def axial_to_cartesian(axial_list):
        pts = np.empty((len(axial_list), 2), dtype=float)
        for i, (q, r) in enumerate(axial_list):
            pts[i, 0] = math.sqrt(3) * (q + r / 2.0)
            pts[i, 1] = 1.5 * r
        return pts

    # Candidate point sets in axial coordinates (11 hexagons each)
    candidates = [
        # Standard hexagonal cluster (7 + 6)
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)],
        # Cross-shaped
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)],
        # 3x3 block + 2
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)],
        # 4x2 + 3
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)],
        # Staircase
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
        # Compact 3x3 + 2
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)],
        # Star with long arms
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)],
        # Elongated
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)],
        # Asymmetric star
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)],
        # Compact with top row
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)],
        # Wide base
        [(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)],
        # Tall narrow
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)],
        # Star with asymmetric outer
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)],
        # Star with diagonal extras
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)],
        # Dense cluster
        [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)],
        # Zigzag
        [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)],
        # Star with top cluster
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)],
        # Trapezoid
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)],
        # Wide top
        [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)],
        # Star with inner diagonal
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)],
        # 4x3 minus corners
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)],
        # Wide with top
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)],
        # Symmetric star
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)],
        # Very tall
        [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)],
        # 3x3 + row
        [(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
        # Dense 4-wide
        [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2)],
        # Star with asymmetric bottom
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)],
        # Tall with top row
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)],
        # Additional diverse
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(0,-1),(3,1)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(1,4),(2,4)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3)],
    ]

    def calc_S_for_beta(pts, beta):
        """Vectorized exact evaluation of minimum enclosing hexagon side length."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6,2)
        projs = pts @ normals.T  # (n_pts, 6)
        D = float(np.max(projs) - np.min(projs)) / 2.0
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        S = (D + APOTHEM) / APOTHEM
        return S, (cx, cy), D

    def smooth_calc_S(pts, beta, cx, cy, temperature=0.1):
        """Smooth surrogate using log-sum-exp for differentiable optimization."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T  # (n_pts, 6)
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        max_val = np.max(all_projs)
        if max_val > 100:
            return float('inf'), (cx, cy), float('inf')
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        S = (D + APOTHEM) / APOTHEM
        return S, (cx, cy), D

    def optimize_shape(pts):
        """Multi-stage optimization: grid, scalar refinement, smooth surrogate, exact polish."""
        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        # Stage 1: Coarse grid search over beta in [0, pi/6]
        n_grid = 49
        for i in range(n_grid):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            beta = i * (math.pi / 6) / (n_grid - 1)
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center

        if best_center is None:
            return float('inf'), 0.0, None

        # Stage 2: Refine beta with bounded scalar optimization
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0],
                                      bounds=(0, math.pi / 6), method='bounded',
                                      options={'maxiter': 300, 'xatol': 1e-11})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        # Stage 3: Smooth surrogate optimization with adaptive temperature
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.3, 0.1, 0.05, 0.02, 0.01]
                start_betas = [best_beta, best_beta + 0.05, best_beta - 0.05, 0.0, math.pi/12]

                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    for sb in start_betas:
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break

                        def smooth_obj(params, t=temp):
                            cx, cy, beta = params
                            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                            S, _, _ = smooth_calc_S(pts, beta, cx, cy, temperature=t)
                            return S

                        init = [best_center[0], best_center[1], sb]
                        res = minimize(smooth_obj, init, method='Nelder-Mead',
                                       options={'maxiter': 400, 'xatol': 1e-11, 'fatol': 1e-11})

                        if res.fun < 1e10:
                            cx_r, cy_r, beta_r = res.x
                            beta_r = ((beta_r % (math.pi/6)) + math.pi/6) % (math.pi/6)
                            S_exact, c_exact, _ = calc_S_for_beta(pts, beta_r)
                            if S_exact < best_S:
                                best_S = S_exact
                                best_beta = beta_r
                                best_center = c_exact
        except Exception:
            pass

        # Stage 4: Final high-precision polish using exact objective
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
                    normals = np.column_stack([np.cos(angles), np.sin(angles)])
                    pts_centered = pts - np.array([cx, cy])
                    projs = pts_centered @ normals.T
                    max_proj = np.max(np.abs(projs))
                    S = (max_proj + APOTHEM) / APOTHEM
                    return S

                polish_starts = [
                    [best_center[0], best_center[1], best_beta],
                    [best_center[0], best_center[1], best_beta + 0.02],
                    [best_center[0], best_center[1], best_beta - 0.02],
                    [best_center[0] + 0.02, best_center[1], best_beta],
                    [best_center[0], best_center[1] + 0.02, best_beta],
                    [best_center[0] - 0.02, best_center[1], best_beta],
                    [best_center[0], best_center[1] - 0.02, best_beta],
                    [best_center[0] + 0.05, best_center[1], best_beta],
                    [best_center[0] - 0.05, best_center[1], best_beta],
                    [best_center[0], best_center[1] + 0.05, best_beta],
                    [best_center[0], best_center[1] - 0.05, best_beta],
                ]

                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                   options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-13})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
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
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        # Fallback: use first candidate with simple centering
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

    # Very small safety scale to ensure no overlap due to floating point
    SCALE = 1.0 + 1e-12
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))

    pts_scaled = np.array(pts_scaled)

    # Compute final side length
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
