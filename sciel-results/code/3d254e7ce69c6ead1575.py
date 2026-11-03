import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar
from itertools import product

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # --- Geometry helpers ---

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def get_normals(beta):
        """Return 6 unit outward normals of the containing hexagon for rotation beta."""
        return np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])

    def compute_D(pts, center, normals):
        """Exact: D = max over points and normals of |(p - c) . n|."""
        centered = pts - np.array(center)
        projs = centered @ normals.T
        return float(np.max(np.abs(projs)))

    def calc_S_for_beta(pts, beta):
        """LP over center for fixed beta. Returns (S, center, D)."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, float('inf')

        normals = get_normals(beta)
        c_obj = np.array([0.0, 0.0, 1.0])
        n_pts = len(pts)

        A_ub_blocks = []
        b_ub_blocks = []
        for nx, ny in normals:
            proj = pts[:, 0] * nx + pts[:, 1] * ny
            A1 = np.tile([nx, ny, -1.0], (n_pts, 1))
            A2 = np.tile([-nx, -ny, -1.0], (n_pts, 1))
            A_ub_blocks.extend([A1, A2])
            b_ub_blocks.extend([proj, -proj])

        A_ub = np.vstack(A_ub_blocks)
        b_ub = np.concatenate(b_ub_blocks)
        bounds = [(None, None), (None, None), (0, None)]

        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = float(res.x[2])
                cx, cy = float(res.x[0]), float(res.x[1])
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, float('inf')
        except Exception:
            return float('inf'), None, float('inf')

    def hexagon_vertices(cx, cy, side, angle_rad):
        """Return 6 vertices of a regular hexagon centered at (cx,cy) with side length and rotation."""
        verts = []
        for k in range(6):
            theta = angle_rad + k * math.pi / 3.0
            verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
        return np.array(verts)

    def point_in_hexagon(px, py, cx, cy, side, angle_rad):
        """Check if point (px,py) is inside the hexagon."""
        # Transform point to hexagon's local frame
        dx = px - cx
        dy = py - cy
        # Rotate by -angle
        ca, sa = math.cos(-angle_rad), math.sin(-angle_rad)
        lx = dx * ca - dy * sa
        ly = dx * sa + dy * ca
        # In local frame, hexagon with side s has apothem a = s*sqrt(3)/2
        # Vertices at angles 0, 60, 120, 180, 240, 300 degrees from center
        # Check 6 half-planes: for each edge, the point must be on the interior side
        # Edge k connects vertex k to vertex k+1
        # Normal to edge k points outward
        # For a hexagon with vertices at angle k*pi/3, the outward normal of edge k (between vert k and vert k+1)
        # is at angle (k + 0.5) * pi/3
        a = side * math.sqrt(3) / 2.0
        for k in range(6):
            theta_n = (k + 0.5) * math.pi / 3.0
            nx = math.cos(theta_n)
            ny = math.sin(theta_n)
            # The edge is at distance a from center along normal (nx, ny)
            # Point must satisfy lx*nx + ly*ny <= a
            if lx * nx + ly * ny > a + 1e-12:
                return False
        return True

    def hexagon_overlap_check(pts, side, angle_rad):
        """Check if any pair of unit hexagons at positions pts overlap.
        Each hexagon has side 1, centered at pts[i], rotated by angle_rad.
        Returns True if any overlap detected."""
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                # Check if any vertex of hex i is inside hex j, or vice versa, or edges cross
                # Simple check: distance between centers must be > 2*inradius for no overlap? No.
                # For two regular hexagons of side 1, the minimum distance between centers
                # for non-overlap depends on relative orientation.
                # Quick reject: if center distance > 2, they can't overlap (diameter is 2)
                dx = pts[i, 0] - pts[j, 0]
                dy = pts[i, 1] - pts[j, 1]
                dist = math.hypot(dx, dy)
                if dist > 2.0 + 1e-9:
                    continue
                # Check vertices of i inside j
                verts_i = hexagon_vertices(pts[i, 0], pts[i, 1], 1.0, angle_rad)
                for vx, vy in verts_i:
                    if point_in_hexagon(vx, vy, pts[j, 0], pts[j, 1], 1.0, angle_rad):
                        return True
                # Check vertices of j inside i
                verts_j = hexagon_vertices(pts[j, 0], pts[j, 1], 1.0, angle_rad)
                for vx, vy in verts_j:
                    if point_in_hexagon(vx, vy, pts[i, 0], pts[i, 1], 1.0, angle_rad):
                        return True
        return False

    def optimize_shape(pts, angle_rad):
        """Optimize center for fixed point set and angle. Returns (S, center)."""
        S, center, D = calc_S_for_beta(pts, angle_rad)
        return S, center

    # --- Generate candidate configurations ---
    # Strategy: Use axial lattice positions but allow continuous perturbation
    # and optimize the center and outer hexagon rotation.

    best_global_S = float('inf')
    best_global_pts = None
    best_global_center = None
    best_global_angle = 0.0

    # Base axial configurations
    base_configs = []

    # Configuration 1: Center + 6 ring + 4 more
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    base_configs.append(axial_1)

    # Configuration 2: 3x4 grid-like
    axial_2 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    base_configs.append(axial_2)

    # Configuration 3: Compact cluster
    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    base_configs.append(axial_3)

    # Configuration 4: Elongated
    axial_4 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    base_configs.append(axial_4)

    # Configuration 5: Another compact
    axial_5 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    base_configs.append(axial_5)

    # Configuration 6: Hexagonal ring + extras
    axial_6 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    base_configs.append(axial_6)

    # Configuration 7: Staggered rows
    axial_7 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    base_configs.append(axial_7)

    # Configuration 8: Dense center
    axial_8 = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    base_configs.append(axial_8)

    # Configuration 9
    axial_9 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    base_configs.append(axial_9)

    # Configuration 10
    axial_10 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    base_configs.append(axial_10)

    # Configuration 11: Symmetric cross
    axial_11 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (-1, 1), (2, -1),
        (1, -1), (-1, -1), (0, 2)
    ]
    base_configs.append(axial_11)

    # Configuration 12
    axial_12 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    base_configs.append(axial_12)

    # Configuration 13: Very compact
    axial_13 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    base_configs.append(axial_13)

    # Configuration 14
    axial_14 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4), (2, 4)
    ]
    base_configs.append(axial_14)

    # Configuration 15
    axial_15 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    base_configs.append(axial_15)

    # For each base config, try multiple outer hexagon rotations and optimize center
    n_rotations = 37  # Try 37 different rotations of the outer hexagon

    for config_idx, axial in enumerate(base_configs):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break

        pts = axial_to_cartesian(axial)

        # Quick bounding box check
        x_range = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
        y_range = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
        rough_S = max(x_range, y_range * 2.0 / math.sqrt(3)) / (2.0 * APOTHEM) + 1.0
        if best_global_S < float('inf') and rough_S > best_global_S * 1.2:
            continue

        # Try multiple outer hexagon rotations
        for rot_idx in range(n_rotations):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            angle_rad = rot_idx * (math.pi / 6) / (n_rotations - 1)

            S, center = optimize_shape(pts, angle_rad)
            if S < best_global_S and center is not None:
                # Verify no overlaps (should be fine for lattice configs, but check)
                if not hexagon_overlap_check(pts, 1.0, angle_rad):
                    best_global_S = S
                    best_global_pts = pts.copy()
                    best_global_center = center
                    best_global_angle = angle_rad

    # --- Local refinement: perturb positions slightly to improve packing ---
    # For the best configuration found, try small perturbations of each hexagon position
    # and re-optimize the center.

    if best_global_pts is not None and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        best_pts = best_global_pts.copy()
        best_angle = best_global_angle
        best_center = best_global_center
        best_S = best_global_S

        # Try perturbing each hexagon in small steps
        n_perturb = 50
        for iter_idx in range(n_perturb):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break

            # Randomly select a hexagon to perturb
            idx = np.random.randint(0, N)
            # Small random perturbation
            dx = np.random.uniform(-0.1, 0.1)
            dy = np.random.uniform(-0.1, 0.1)

            new_pts = best_pts.copy()
            new_pts[idx, 0] += dx
            new_pts[idx, 1] += dy

            # Check for overlaps
            if hexagon_overlap_check(new_pts, 1.0, best_angle):
                continue

            # Re-optimize center
            S_new, center_new = optimize_shape(new_pts, best_angle)
            if S_new < best_S - 1e-10 and center_new is not None:
                best_S = S_new
                best_pts = new_pts
                best_center = center_new

        if best_S < best_global_S:
            best_global_S = best_S
            best_global_pts = best_pts
            best_global_center = best_center

    # --- Final: also try optimizing the outer hexagon rotation more finely around best ---
    if best_global_pts is not None and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        pts = best_global_pts
        best_S_local = best_global_S
        best_angle_local = best_global_angle
        best_center_local = best_global_center

        # Fine grid around best angle
        for i in range(20):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            angle = best_angle_local + (i - 10) * 0.01
            angle = ((angle % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
            S, center = optimize_shape(pts, angle)
            if S < best_S_local and center is not None:
                if not hexagon_overlap_check(pts, 1.0, angle):
                    best_S_local = S
                    best_angle_local = angle
                    best_center_local = center

        if best_S_local < best_global_S:
            best_global_S = best_S_local
            best_global_angle = best_angle_local
            best_global_center = best_center_local

    # --- Fallback ---
    if best_global_center is None or best_global_S == float('inf'):
        pts = axial_to_cartesian(base_configs[0])
        best_global_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        best_global_angle = 0.0
        normals = get_normals(best_global_angle)
        max_proj = 0.0
        for nx, ny in normals:
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
        best_global_S = (max_proj + APOTHEM) / APOTHEM
        best_global_pts = pts

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_angle)

    # --- Final: recompute exact S with the chosen (center, beta) ---
    pts_scaled = best_global_pts  # No scaling needed if we trust the LP

    normals = get_normals(best_global_angle)
    max_proj = 0.0
    for nx, ny in normals:
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
