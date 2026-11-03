import math
import numpy as np
import time
from scipy.optimize import linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 145.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Axial coordinates for the 11 hexagons in a tight 3x3 hexagonal block
    # The 3x3 hex block has 19 cells, we pick 11 that form a compact shape
    # Centered configuration
    base_config = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (1, 1), (-1, 1),
        (0, -1), (1, -1),
        (0, 2), (-1, 2), (2, 0)
    ]

    # Alternative compact configurations
    config_2 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (-1, 1), (1, -1), (-1, -1),
        (0, 2), (0, -2)
    ]

    config_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]

    config_4 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]

    configs = [base_config, config_2, config_3, config_4]

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def compute_S_and_center(pts, beta):
        """Compute minimal enclosing hexagon side length and center."""
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
                return S, (cx, cy)
            else:
                # Fallback: use centroid
                cx = float(np.mean(pts[:, 0]))
                cy = float(np.mean(pts[:, 1]))
                max_proj = 0
                for nx, ny in normals:
                    projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
                    local_max = max(np.max(projs), -np.min(projs))
                    if local_max > max_proj:
                        max_proj = local_max
                S = (max_proj + APOTHEM) / APOTHEM
                return S, (cx, cy)
        except Exception:
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))
            max_proj = 0
            for nx, ny in normals:
                projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
                local_max = max(np.max(projs), -np.min(projs))
                if local_max > max_proj:
                    max_proj = local_max
            S = (max_proj + APOTHEM) / APOTHEM
            return S, (cx, cy)

    def check_disjoint(pts):
        """Check if hexagons at given Cartesian positions are disjoint (unit hexagons, same orientation)."""
        # For two unit hexagons with same orientation, they overlap iff
        # the distance between centers is less than sqrt(3) in all 6 normal directions
        normals = np.array([
            [math.cos(math.pi/6 + k*math.pi/3), math.sin(math.pi/6 + k*math.pi/3)]
            for k in range(6)
        ])
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                d = pts[i] - pts[j]
                projs = np.abs(d @ normals.T)
                if np.min(projs) < math.sqrt(3) - 1e-9:
                    return False
        return True

    best_S = float('inf')
    best_beta = 0.0
    best_center = None
    best_pts = None

    for cfg in configs:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break

        pts = axial_to_cartesian(cfg)

        if not check_disjoint(pts):
            continue

        # Try multiple rotation angles
        for i in range(12):
            beta = i * (math.pi / 6) / 12
            S, center = compute_S_and_center(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
                best_pts = pts.copy()

        # Fine-tune around best angle found for this config
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break

    # Final refinement: try small perturbations of beta around best
    if best_pts is not None and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        for delta_beta in [0.001, -0.001, 0.005, -0.005, 0.01, -0.01]:
            beta_try = best_beta + delta_beta
            S, center = compute_S_and_center(best_pts, beta_try)
            if S < best_S:
                best_S = S
                best_beta = beta_try
                best_center = center

    # Fallback
    if best_pts is None:
        best_pts = axial_to_cartesian(base_config)
        best_center = (float(np.mean(best_pts[:, 0])), float(np.mean(best_pts[:, 1])))
        best_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = (best_pts[:, 0] - best_center[0]) * nx + (best_pts[:, 1] - best_center[1]) * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_center
    angle_deg = math.degrees(best_beta)

    # Compute final side length
    normals = []
    for k in range(6):
        ang = best_beta + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))

    max_proj = 0
    for nx, ny in normals:
        projs = (best_pts[:, 0] - cx) * nx + (best_pts[:, 1] - cy) * ny
        local_max = max(np.max(projs), -np.min(projs))
        if local_max > max_proj:
            max_proj = local_max

    S_final = (max_proj + APOTHEM) / APOTHEM

    inner = []
    for x, y in best_pts:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)

    return inner, center, side, rotation
