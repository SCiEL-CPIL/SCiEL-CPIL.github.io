import math
import numpy as np
import time
from scipy.optimize import linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Hexagon vertices for side length 1 (pointy-top, 0° rotation)
    hex_verts = np.array([
        [math.cos(math.radians(30 + 60 * k)), math.sin(math.radians(30 + 60 * k))]
        for k in range(6)
    ])

    # Support function values for the Minkowski sum H+(-H) in 6 directions
    # For a regular hexagon of side 1, H+(-H) is a regular hexagon of side 2
    # Support in direction theta: 2 * |cos(theta - 30°)| for the 6 edge normals
    # Actually for H+(-H) with side 2: apothem = 2*sqrt(3)/2 = sqrt(3)
    # Support in direction (nx,ny): 2*max(|dot(v, n)|) over vertices
    # Simpler: the Minkowski sum of two unit hexagons (same orientation) is a hexagon
    # with side length 2. Its support in direction n is:
    #   h(n) = 2 * max_{v in hex_verts} dot(v, n)  ... no that's not right either.
    # 
    # For two convex polygons A and B, h_{A+B}(n) = h_A(n) + h_B(n)
    # h_H(n) for unit hexagon: max over 6 vertices of dot(v, n)
    # For H + (-H): h(n) = h_H(n) + h_H(-n) = h_H(n) + h_H(n) = 2*h_H(n)
    # (since hexagon is centrally symmetric)
    #
    # So the non-overlap constraint between hex i at position p_i and hex j at p_j:
    #   |dot(p_i - p_j, n_k)| <= h_H(n_k) for all 6 directions n_k
    # where n_k are the 6 edge-normal directions of the hexagon.
    #
    # Edge normals for a regular hexagon (pointy-top):
    #   angles: 0°, 60°, 120°, 180°, 240°, 300°  (or equivalently 30°, 90°, ... for flat-top)
    # Let's use the 6 directions perpendicular to edges.
    # For pointy-top hex with vertices at 30°, 90°, 150°, 210°, 270°, 330°:
    # Edge normals are at 0°, 60°, 120°, 180°, 240°, 300°
    
    edge_normals = np.array([
        [math.cos(math.radians(60 * k)), math.sin(math.radians(60 * k))]
        for k in range(6)
    ])
    
    # h_H(n) = max dot(v, n) over vertices
    h_vals = np.max(hex_verts @ edge_normals.T, axis=0)  # 6 values
    # For non-overlap: |dot(p_i - p_j, n_k)| <= h_vals[k] for all k
    # Minkowski sum support: h_sum(n_k) = 2 * h_vals[k]
    # Constraint: dot(p_i - p_j, n_k) <= h_sum(n_k) AND dot(p_j - p_i, n_k) <= h_sum(n_k)
    # i.e., |dot(p_i - p_j, n_k)| <= 2 * h_vals[k]
    
    mink_support = 2.0 * h_vals  # 6 values

    # Outer hexagon: center (cx, cy), side S, rotation beta
    # Inner hex i at (x_i, y_i) with 0° rotation must be inside outer hex.
    # Outer hex has vertices at distance S from center, rotated by beta.
    # The 6 edge normals of the outer hex are at angles: beta + 30° + 60°*k
    # Wait, for a regular hexagon with side S, the vertices are at distance S from center.
    # Edge normals (outward) are at angles: beta + 0°, beta + 60°, ..., beta + 300°
    # Actually depends on orientation. Let's define:
    # Outer hex vertices: center + S * [cos(beta + 60k), sin(beta + 60k)] for k=0..5
    # Edge normals (perpendicular to edges, pointing outward):
    # For the edge between vertex k and vertex k+1, the normal is at angle beta + 30° + 60k
    # 
    # A point p is inside the outer hex iff for all 6 outward normals n_k:
    #   dot(p - center, n_k) <= S * cos(30°) = S * sqrt(3)/2 = S * APOTHEM
    # 
    # But we also need the inner hexagon (not just its center) to be inside.
    # The inner hex at position p_i with 0° rotation: its support in direction n_k is
    #   dot(p_i, n_k) + h_H(n_k)
    # where h_H(n_k) = max dot(v, n_k) over inner hex vertices.
    #
    # So constraint: dot(p_i, n_k) + h_H(n_k) <= S * APOTHEM + dot(center, n_k)
    # i.e., dot(p_i - center, n_k) <= S * APOTHEM - h_H(n_k)
    #
    # For the inner hexagon to be fully inside, we need this for all 6 outer normals.
    # But wait - the inner hexagon has its own 6 edge normals. The correct condition is:
    # The inner hex is inside the outer hex iff for every direction, the support of the
    # inner hex (relative to outer center) is <= the support of the outer hex.
    # 
    # For regular hexagons, it suffices to check the 6 edge-normal directions of the
    # OUTER hexagon, because the inner hexagon's support function is convex and piecewise
    # linear, and the outer hexagon's support is also piecewise linear. The critical
    # directions are the vertices of the outer hexagon's normal fan, which are the 6
    # edge normals.
    #
    # Actually, I need to be more careful. The inner hexagon must satisfy:
    #   h_inner(p_i, n) <= h_outer(center, S, beta, n) for all n
    # h_inner(p_i, n) = dot(p_i, n) + h_H(n)  (where h_H is support of unit hex at origin)
    # h_outer(center, S, beta, n) = dot(center, n) + S * h_unit_hex_rotated(beta, n)
    # where h_unit_hex_rotated(beta, n) is support of unit hex rotated by beta.
    #
    # For a regular hexagon, h(n) depends only on the angle of n modulo 60°.
    # h_unit(n) = max_k dot(v_k, n) where v_k are vertices at 30°+60°k.
    # h_unit(n) = cos(30° - (angle(n) mod 60°)) ... let me just compute numerically.
    #
    # For the LP formulation, I'll fix beta and solve for positions + S.
    # For a fixed beta, the outer hex normals are known, and the constraints are linear
    # in (x_i, y_i, cx, cy, S).

    def solve_lp_for_beta(beta, init_positions=None):
        """Solve LP for fixed outer rotation beta.
        Variables: x_0..x_10, y_0..y_10, cx, cy, S  (24 variables)
        Minimize: S
        Constraints:
          1. Non-overlap: for each pair (i,j) and each of 6 directions:
             dot(p_i - p_j, n_k) <= mink_support[k]
             dot(p_j - p_i, n_k) <= mink_support[k]
          2. Containment: for each inner hex i and each of 6 outer normals:
             dot(p_i - center, outer_n_k) + h_H(outer_n_k) <= S * APOTHEM
        """
        n_vars = 2 * N + 3  # x_i, y_i for i=0..N-1, then cx, cy, S
        idx = {}
        vi = 0
        for i in range(N):
            idx[f'x{i}'] = vi; vi += 1
            idx[f'y{i}'] = vi; vi += 1
        idx['cx'] = vi; vi += 1
        idx['cy'] = vi; vi += 1
        idx['S'] = vi; vi += 1

        # Objective: minimize S
        c_obj = np.zeros(n_vars)
        c_obj[idx['S']] = 1.0

        A_ub = []
        b_ub = []

        # Outer hex normals for rotation beta
        outer_normals = np.array([
            [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
            for k in range(6)
        ])

        # h_H(n) for each outer normal direction
        h_inner_supports = np.max(hex_verts @ outer_normals.T, axis=0)

        # Containment constraints:
        # dot(p_i - center, outer_n_k) <= S * APOTHEM - h_inner_supports[k]
        # dot(p_i, outer_n_k) - dot(center, outer_n_k) - S * APOTHEM <= -h_inner_supports[k]
        for i in range(N):
            for k in range(6):
                nx, ny = outer_normals[k]
                row = np.zeros(n_vars)
                row[idx[f'x{i}']] = nx
                row[idx[f'y{i}']] = ny
                row[idx['cx']] = -nx
                row[idx['cy']] = -ny
                row[idx['S']] = -APOTHEM
                A_ub.append(row)
                b_ub.append(-h_inner_supports[k])

        # Non-overlap constraints:
        # For pair (i, j), direction k:
        # dot(p_i - p_j, n_k) <= mink_support[k]
        # dot(p_j - p_i, n_k) <= mink_support[k]
        for i in range(N):
            for j in range(i + 1, N):
                for k in range(6):
                    nx, ny = edge_normals[k]
                    row1 = np.zeros(n_vars)
                    row1[idx[f'x{i}']] = nx
                    row1[idx[f'y{i}']] = ny
                    row1[idx[f'x{j}']] = -nx
                    row1[idx[f'y{j}']] = -ny
                    A_ub.append(row1)
                    b_ub.append(mink_support[k])

                    row2 = np.zeros(n_vars)
                    row2[idx[f'x{i}']] = -nx
                    row2[idx[f'y{i}']] = -ny
                    row2[idx[f'x{j}']] = nx
                    row2[idx[f'y{j}']] = ny
                    A_ub.append(row2)
                    b_ub.append(mink_support[k])

        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None)] * n_vars

        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                positions = np.array([
                    [res.x[idx[f'x{i}']], res.x[idx[f'y{i}']]]
                    for i in range(N)
                ])
                cx = res.x[idx['cx']]
                cy = res.x[idx['cy']]
                S = res.x[idx['S']]
                return S, positions, (cx, cy)
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    # Initial positions: hexagonal lattice
    def hex_lattice_positions():
        pts = []
        # 3-4-3-1 or similar arrangement
        # Use a compact hexagonal arrangement
        axial = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (-2, 0), (0, 2), (1, 2)
        ]
        for q, r in axial:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append([x, y])
        return np.array(pts)

    # Try multiple outer rotation angles
    best_S = float('inf')
    best_positions = None
    best_center = None
    best_beta = 0.0

    # Coarse search over beta
    for i in range(13):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        beta = i * (math.pi / 6.0) / 12.0
        S, pos, center = solve_lp_for_beta(beta)
        if S < best_S:
            best_S = S
            best_positions = pos
            best_center = center
            best_beta = beta

    # Refine around best beta
    if best_positions is not None:
        for i in range(13):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            beta = best_beta + (i - 6) * (math.pi / 6.0) / 144.0
            if beta < 0 or beta >= math.pi / 6.0:
                continue
            S, pos, center = solve_lp_for_beta(beta)
            if S < best_S:
                best_S = S
                best_positions = pos
                best_center = center
                best_beta = beta

    # If LP failed, fall back to the original lattice approach
    if best_positions is None or best_S == float('inf'):
        axial = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (1, 1), (-1, 2), (-2, 1), (2, -1)
        ]
        best_positions = np.array([
            [math.sqrt(3) * (q + r / 2.0), 1.5 * r] for q, r in axial
        ])
        best_center = (0.0, 0.0)
        best_beta = 0.0
        # Compute S for this configuration
        outer_normals = np.array([
            [math.cos(math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(math.pi / 6.0 + k * math.pi / 3.0)]
            for k in range(6)
        ])
        h_inner = np.max(hex_verts @ outer_normals.T, axis=0)
        max_proj = 0
        for k in range(6):
            nx, ny = outer_normals[k]
            projs = best_positions[:, 0] * nx + best_positions[:, 1] * ny
            max_proj = max(max_proj, np.max(projs) - best_center[0] * nx - best_center[1] * ny)
        best_S = (max_proj - np.min(h_inner)) / APOTHEM  # approximate
        # More careful:
        max_proj = 0
        for k in range(6):
            nx, ny = outer_normals[k]
            projs = (best_positions[:, 0] - best_center[0]) * nx + (best_positions[:, 1] - best_center[1]) * ny
            max_proj = max(max_proj, np.max(projs))
        best_S = (max_proj + APOTHEM) / APOTHEM

    # Build output
    cx, cy = best_center
    angle_deg = math.degrees(best_beta)

    # Small scale factor for safety
    SCALE = 1.0 + 1e-10
    pts_scaled = best_positions.copy()
    pts_scaled[:, 0] = cx + (pts_scaled[:, 0] - cx) * SCALE
    pts_scaled[:, 1] = cy + (pts_scaled[:, 1] - cy) * SCALE

    # Recompute S with scaled positions
    outer_normals = np.array([
        [math.cos(best_beta + math.pi / 6.0 + k * math.pi / 3.0),
         math.sin(best_beta + math.pi / 6.0 + k * math.pi / 3.0)]
        for k in range(6)
    ])
    h_inner = np.max(hex_verts @ outer_normals.T, axis=0)
    max_proj = 0
    for k in range(6):
        nx, ny = outer_normals[k]
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs))
    S_final = (max_proj + APOTHEM) / APOTHEM

    inner = []
    for i in range(N):
        inner.append([float(pts_scaled[i, 0]), float(pts_scaled[i, 1]), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)

    return inner, center, side, rotation
