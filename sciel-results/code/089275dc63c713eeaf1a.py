import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 12.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate candidate configurations
    candidates = []

    # Configuration 1: Central hexagon with 6 around + 4 more
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (0, -2)
    ]
    candidates.append(axial_1)

    # Configuration 2: Linear arrangement
    axial_2 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1)
    ]
    candidates.append(axial_2)

    # Configuration 3: Compact cluster
    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_3)

    # Configuration 4: Diamond shape
    axial_4 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2),
        (1, 1), (-1, -1)
    ]
    candidates.append(axial_4)

    # Configuration 5: Staggered rows
    axial_5 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_5)

    # Configuration 6: Another compact
    axial_6 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_6)

    # Configuration 7: Hexagonal ring
    axial_7 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3), (-2, 2), (-2, 1)
    ]
    candidates.append(axial_7)

    # Configuration 8: Triangular
    axial_8 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_8)

    # Configuration 9: Asymmetric
    axial_9 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_9)

    # Configuration 10: Cross
    axial_10 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2),
        (1, 1), (-1, -1)
    ]
    candidates.append(axial_10)

    # Configuration 11: Elongated
    axial_11 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append(axial_11)

    # Configuration 12: Mixed
    axial_12 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_12)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def hexagon_vertices(cx, cy, S, angle_deg):
        """Return 6 vertices of a regular hexagon with side S, center (cx,cy), rotated by angle_deg."""
        verts = []
        for k in range(6):
            ang = math.radians(angle_deg) + k * math.pi / 3.0
            x = cx + S * math.cos(ang)
            y = cy + S * math.sin(ang)
            verts.append((x, y))
        return verts

    def polygon_overlap(v1, v2):
        """Check if two convex polygons overlap using separating axis theorem."""
        def get_edges(verts):
            edges = []
            n = len(verts)
            for i in range(n):
                x1, y1 = verts[i]
                x2, y2 = verts[(i + 1) % n]
                dx, dy = x2 - x1, y2 - y1
                length = math.sqrt(dx * dx + dy * dy)
                if length > 1e-10:
                    nx, ny = -dy / length, dx / length
                    edges.append((nx, ny))
            return edges

        def project(verts, axis):
            ax, ay = axis
            projections = [x * ax + y * ay for x, y in verts]
            return min(projections), max(projections)

        edges = get_edges(v1) + get_edges(v2)
        for nx, ny in edges:
            min1, max1 = project(v1, (nx, ny))
            min2, max2 = project(v2, (nx, ny))
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return False
        return True

    def check_validity(inner_pts, outer_center, outer_S, outer_angle):
        """Check if inner hexagons are disjoint and inside the outer hexagon."""
        outer_verts = hexagon_vertices(outer_center[0], outer_center[1], outer_S, outer_angle)
        
        # Check each inner hexagon is inside outer
        for i, (x, y, ang) in enumerate(inner_pts):
            inner_verts = hexagon_vertices(x, y, 1.0, ang)
            # Check all vertices of inner are inside outer
            for vx, vy in inner_verts:
                # Point in hexagon test using cross products
                inside = True
                n = len(outer_verts)
                for j in range(n):
                    x1, y1 = outer_verts[j]
                    x2, y2 = outer_verts[(j + 1) % n]
                    cross = (x2 - x1) * (vy - y1) - (y2 - y1) * (vx - x1)
                    if cross < -1e-9:
                        inside = False
                        break
                if not inside:
                    return False
        
        # Check inner hexagons are disjoint
        for i in range(len(inner_pts)):
            for j in range(i + 1, len(inner_pts)):
                vi = hexagon_vertices(inner_pts[i][0], inner_pts[i][1], 1.0, inner_pts[i][2])
                vj = hexagon_vertices(inner_pts[j][0], inner_pts[j][1], 1.0, inner_pts[j][2])
                if polygon_overlap(vi, vj):
                    return False
        return True

    def calc_S_for_beta(pts, beta):
        """Find minimum enclosing hexagon (rotated by beta) for given points."""
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        
        # Linear programming: min D s.t. all points inside hexagon
        # Variables: cx, cy, D
        # Constraints: for each point and each normal, |p·n - c·n| <= D
        
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for px, py in pts:
                # (px-cx)*nx + (py-cy)*ny <= D
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                # -(px-cx)*nx - (py-cy)*ny <= D
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            from scipy.optimize import linprog
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

    def continuous_optimize(pts, beta_init, center_init):
        """Use continuous optimization to refine the outer hexagon parameters."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        
        def objective(params):
            cx, cy, beta = params
            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
            S, _, _ = calc_S_for_beta(pts, beta)
            if S == float('inf'):
                return 1e10
            return S
        
        init = [center_init[0], center_init[1], beta_init]
        
        try:
            res = minimize(objective, init, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < 1e10:
                cx_r, cy_r, beta_r = res.x
                beta_r = ((beta_r % (math.pi/6)) + math.pi/6) % (math.pi/6)
                S_exact, c_exact, _ = calc_S_for_beta(pts, beta_r)
                return S_exact, c_exact, beta_r
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    # Sort candidates by quick estimate
    def quick_estimate(axial_list):
        pts = axial_to_cartesian(axial_list)
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        return (max_proj + APOTHEM) / APOTHEM

    estimated_scores = [(quick_estimate(axial), i) for i, axial in enumerate(candidates)]
    estimated_scores.sort()
    
    # Process candidates
    for _, idx in estimated_scores:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        axial = candidates[idx]
        pts = axial_to_cartesian(axial)
        
        # Coarse grid search over beta
        best_S_for_pts = float('inf')
        best_beta_for_pts = 0.0
        best_center_for_pts = None
        
        for i in range(61):
            beta = i * (math.pi / 6) / 60
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S_for_pts:
                best_S_for_pts = S
                best_beta_for_pts = beta
                best_center_for_pts = center
        
        # Refine with continuous optimization
        if best_center_for_pts is not None:
            S_refined, center_refined, beta_refined = continuous_optimize(
                pts, best_beta_for_pts, best_center_for_pts
            )
            if S_refined < best_S_for_pts:
                best_S_for_pts = S_refined
                best_beta_for_pts = beta_refined
                best_center_for_pts = center_refined
        
        if best_S_for_pts < best_global_S:
            best_global_S = best_S_for_pts
            best_global_pts = pts
            best_global_beta = best_beta_for_pts
            best_global_center = best_center_for_pts

    # Fallback if no valid solution found
    if best_global_center is None or best_global_S == float('inf'):
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
    
    # Scale points slightly to ensure no overlap
    SCALE = 1.0 + 1e-10
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
