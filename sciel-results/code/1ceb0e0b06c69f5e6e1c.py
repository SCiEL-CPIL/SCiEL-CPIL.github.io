import math
import numpy as np
import time
from scipy.optimize import minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Define candidate point sets in axial coordinates for unit hexagon centers
    # Each list has exactly N=11 points
    candidates = []

    # Compact 3x4 rectangular-ish block
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ])

    # 4x3 rectangular-ish block
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ])

    # Hexagonal cluster (center + 6 ring + 4 more)
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ])

    # Another hexagonal variant
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ])

    # Elongated strip
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ])

    # Compact 3x3 + 2
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ])

    # Diamond-ish
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ])

    # Asymmetric compact
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ])

    # Another asymmetric
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ])

    # Compact with offset
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ])

    # Hexagonal with extra
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ])

    # Compact 4-wide
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ])

    # Very compact
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1), (0, -1)
    ])

    # 3x4 variant
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ])

    # Compact 3x3+2 variant
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4)
    ])

    # 3x4 another variant
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ])

    # 4x3 variant
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3)
    ])

    # Hexagonal variant
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, -2)
    ])

    # 3x4 variant
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4), (2, 4)
    ])

    # 4x3 variant
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2)
    ])

    # Compact
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ])

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def support_function(pts, beta):
        """Compute the maximum projection of pts in all 6 hexagon normal directions."""
        cx0 = np.mean(pts[:, 0])
        cy0 = np.mean(pts[:, 1])
        pts_centered = pts - np.array([cx0, cy0])

        max_proj = 0.0
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = pts_centered[:, 0] * nx + pts_centered[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))

        return max_proj, (cx0, cy0)

    def min_enclosing_hex_side(pts):
        """Find the minimal enclosing regular hexagon side length for pts."""
        best_L = float('inf')
        best_beta = 0.0
        best_center = None

        # Coarse grid search over beta in [0, pi/6]
        for i in range(73):
            beta = i * (math.pi / 6) / 72
            D, center = support_function(pts, beta)
            # Side length of enclosing hexagon: D is max projection from center
            # For a regular hexagon with side S, apothem = S * sqrt(3)/2
            # The max projection from center to a vertex is S
            # Actually, for a regular hexagon centered at origin with side S,
            # the support function in direction normal to a face is S * sqrt(3)/2 (apothem)
            # The max projection of the point set in that direction must be <= apothem
            # So we need D + apothem_unit <= apothem_outer
            # D is the max projection of points from their centroid
            # The enclosing hexagon must contain all points, so:
            # D + APOTHEM (unit hexagon apothem) <= S * APOTHEM (outer hexagon apothem)
            # S >= (D + APOTHEM) / APOTHEM
            L = (D + APOTHEM) / APOTHEM
            if L < best_L:
                best_L = L
                best_beta = beta
                best_center = center

        # Refine with bounded optimization
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def obj(beta):
                    D, _ = support_function(pts, beta)
                    return (D + APOTHEM) / APOTHEM

                res = minimize_scalar(obj, bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 500, 'xatol': 1e-14})
                if res.fun < best_L:
                    best_L = res.fun
                    best_beta = res.x
                    _, best_center = support_function(pts, best_beta)
        except Exception:
            pass

        return best_L, best_beta, best_center

    best_S = float('inf')
    best_pts = None
    best_beta = 0.0
    best_center = None

    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        L, beta, center = min_enclosing_hex_side(pts)
        if L < best_S:
            best_S = L
            best_pts = pts
            best_beta = beta
            best_center = center

    if best_pts is None:
        best_pts = axial_to_cartesian(candidates[0])
        best_center = (float(np.mean(best_pts[:, 0])), float(np.mean(best_pts[:, 1])))
        best_beta = 0.0
        D, _ = support_function(best_pts, 0.0)
        best_S = (D + APOTHEM) / APOTHEM

    cx, cy = best_center
    angle_deg = math.degrees(best_beta)

    inner = []
    for x, y in best_pts:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(best_S)
    rotation = float(angle_deg)

    return inner, center, side, rotation
