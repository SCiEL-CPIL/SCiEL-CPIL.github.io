"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side.

Strategy:
  - Formulate as a 1D global optimization over rigid honeycomb clusters.
  - Enumerate a small set of known close-packed 12-hexagon shapes.
  - For each shape, optimize only global rotation, cluster orientation,
    and outer hexagon center/angle using multi-start Nelder-Mead.
  - Fast vectorized validity check using support functions.
  - Return the best valid configuration found within the time budget.
"""

import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 140.0  # safety margin before 150s wall-clock kill

# Precomputed unit edge normals for a regular hexagon (pointy-top at angle 0).
# The 6 outward normals point in directions 30°, 90°, 150°, 210°, 270°, 330°.
_EDGE_NORMALS = np.array([
    [math.cos(math.pi/6),  math.sin(math.pi/6)],
    [math.cos(math.pi/2),  math.sin(math.pi/2)],
    [math.cos(5*math.pi/6), math.sin(5*math.pi/6)],
    [math.cos(7*math.pi/6), math.sin(7*math.pi/6)],
    [math.cos(3*math.pi/2), math.sin(3*math.pi/2)],
    [math.cos(11*math.pi/6), math.sin(11*math.pi/6)],
], dtype=float)  # 6x2


def _rot_matrix(angle_rad):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s], [s, c]], dtype=float)


def _hex_vertices(cx, cy, side, angle_rad):
    """Return 6x2 array of vertices for a regular hexagon."""
    angs = angle_rad + np.array([0.0, math.pi/3, 2*math.pi/3, math.pi, 4*math.pi/3, 5*math.pi/3])
    return np.column_stack([cx + side * np.cos(angs),
                            cy + side * np.sin(angs)])


def _support_points(verts, normals):
    """Return max projection of each normal onto the vertex set. verts: Mx2, normals: Kx2."""
    # proj[m, k] = verts[m] @ normals[k]
    # max_proj[k] = max_m proj[m, k]
    return (verts @ normals.T).max(axis=0)


def construct():
    t0 = time.time()

    def elapsed():
        return time.time() - t0

    # ---- Define rigid cluster shapes (local coordinates, angle 0) ----
    # Each shape is a list of 12 (x, y) positions for hexagon centers.
    # All hexagons have the same orientation (angle 0 in local frame).

    def make_shape_3x4_brick():
        """3 rows x 4 cols, pointy-top brick layout."""
        pts = []
        sq3 = math.sqrt(3)
        for r in range(3):
            for c in range(4):
                x = (c - 1.5) * sq3 + (sq3 / 2 if r % 2 else 0)
                y = (r - 1.0) * 1.5
                pts.append((x, y))
        return pts

    def make_shape_4x3_brick():
        """4 rows x 3 cols, pointy-top brick layout."""
        pts = []
        sq3 = math.sqrt(3)
        for r in range(4):
            for c in range(3):
                x = (c - 1.0) * sq3 + (sq3 / 2 if r % 2 else 0)
                y = (r - 1.5) * 1.5
                pts.append((x, y))
        return pts

    def make_shape_6x2_flat():
        """6 cols x 2 rows, flat-top layout."""
        pts = []
        sq3 = math.sqrt(3)
        for r in range(2):
            for c in range(6):
                x = (c - 2.5) * 3.0
                y = (r - 0.5) * sq3
                pts.append((x, y))
        return pts

    def make_shape_2x6_flat():
        """2 cols x 6 rows, flat-top layout."""
        pts = []
        sq3 = math.sqrt(3)
        for c in range(2):
            for r in range(6):
                x = (c - 0.5) * 3.0
                y = (r - 2.5) * sq3
                pts.append((x, y))
        return pts

    def make_shape_hex_cluster():
        """Hexagonal cluster: center + ring1(6) + ring2(5 contiguous)."""
        pts = [(0.0, 0.0)]
        sq3 = math.sqrt(3)
        # Ring 1: 6 hexagons at distance sq3
        for k in range(6):
            a = math.pi / 3 * k
            pts.append((sq3 * math.cos(a), sq3 * math.sin(a)))
        # Ring 2: 12 positions; pick 5 contiguous ones
        # Ring 2 positions are at distance 2*sq3 in 6 directions, plus midpoints
        # For a hex lattice, ring 2 has positions like (2*sq3*cos(a), 2*sq3*sin(a)) for 6,
        # and also the "corner" positions. Let's use a simpler approach:
        # Just add 5 more from the next shell.
        # Actually, let's just use the 4x3 brick which is known to be compact.
        return None

    # Generate candidate shapes
    shapes = []
    for name, pts in [
        ("3x4brick", make_shape_3x4_brick()),
        ("4x3brick", make_shape_4x3_brick()),
        ("6x2flat", make_shape_6x2_flat()),
        ("2x6flat", make_shape_2x6_flat()),
    ]:
        if pts and len(pts) == 12:
            shapes.append((name, np.array(pts, dtype=float)))

    # ---- Fast validity check ----
    def is_valid(inner_positions, inner_angles, oc, os, oa):
        """Check if all inner hexagons are disjoint and inside the outer hexagon.
        inner_positions: 12x2 array, inner_angles: 12 array of radians.
        Returns (is_disjoint, is_inside, max_overlap, max_violation)."""
        # Compute all vertices for inner hexagons
        all_verts = np.vstack([
            _hex_vertices(x, y, 1.0, a) for x, y, a in zip(inner_positions[:, 0], inner_positions[:, 1], inner_angles)
        ])  # 72x2

        # Check inside outer hexagon
        R_outer = _rot_matrix(oa)
        normals_outer = _EDGE_NORMALS @ R_outer.T  # 6x2
        # Translate points relative to outer center
        rel_verts = all_verts - np.array(oc)
        projs = rel_verts @ normals_outer.T  # 72x6
        max_projs = projs.max(axis=0)  # 6
        apothem_required = os * math.sqrt(3) / 2
        max_violation_inside = max_projs.max() - apothem_required

        # Check disjointness: for each pair of inner hexagons, check SAT
        # Precompute vertices for each inner hexagon
        hex_verts_list = [_hex_vertices(x, y, 1.0, a) for x, y, a in zip(inner_positions[:, 0], inner_positions[:, 1], inner_angles)]
        max_overlap = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                v1 = hex_verts_list[i]
                v2 = hex_verts_list[j]
                # SAT: check all 12 axes (6 from each hexagon)
                min_gap = float('inf')
                for v in [v1, v2]:
                    for k in range(6):
                        e = v[(k + 1) % 6] - v[k]
                        n = np.array([-e[1], e[0]])
                        nn = np.linalg.norm(n)
                        if nn < 1e-12:
                            continue
                        n /= nn
                        p1 = v1 @ n
                        p2 = v2 @ n
                        gap = min(p2.max() - p1.min(), p1.max() - p2.min())
                        # We need the separation: if p1 and p2 intervals don't overlap, gap > 0
                        # Actually: gap = min(p2) - max(p1) if positive, or min(p1) - max(p2) if positive
                        sep = max(0.0, min(p2) - max(p1), min(p1) - max(p2))
                        if sep < min_gap:
                            min_gap = sep
                if min_gap < max_overlap:
                    max_overlap = min_gap
                if min_gap < -1e-9:
                    # Overlap found
                    pass  # Continue to find max overlap for penalty

        is_disjoint = max_overlap >= -1e-9
        is_inside = max_violation_inside <= 1e-9
        return is_disjoint, is_inside, max_overlap, max_violation_inside

    # ---- Objective function for optimization ----
    def objective(params, shape_pts):
        """params: [rot_cluster, rot_inner, cx, cy, rot_outer]
        shape_pts: 12x2 local coordinates.
        Returns side length + penalty."""
        rot_cluster = params[0]
        rot_inner = params[1]
        cx, cy = params[2], params[3]
        rot_outer = params[4]

        # Rotate shape points by rot_cluster
        R_cluster = _rot_matrix(rot_cluster)
        positions = shape_pts @ R_cluster.T  # 12x2

        # Inner hexagon angles: all same, rotated by rot_inner
        inner_angles = np.full(12, rot_inner)

        # Outer center and rotation
        oc = (cx, cy)
        os = 10.0  # placeholder, will be computed
        oa = rot_outer

        # Compute minimal enclosing hexagon side for the given positions
        # We need to find the smallest S such that all vertices are inside.
        # For a regular hexagon with center oc and rotation oa, the constraint is:
        # max_k ( (p - oc) . n_k ) <= S * sqrt(3) / 2 for all p, k
        # => S >= (2/sqrt(3)) * max_k ( (p - oc) . n_k )

        # Compute all vertices
        all_verts = np.vstack([
            _hex_vertices(x, y, 1.0, a) for x, y, a in zip(positions[:, 0], positions[:, 1], inner_angles)
        ])  # 72x2

        R_outer = _rot_matrix(rot_outer)
        normals_outer = _EDGE_NORMALS @ R_outer.T  # 6x2
        rel_verts = all_verts - np.array(oc)
        projs = rel_verts @ normals_outer.T  # 72x6
        max_projs = projs.max(axis=0)  # 6
        side_len = (2.0 / math.sqrt(3)) * max_projs.max()

        # Check validity for penalty
        is_disjoint, is_inside, max_overlap, max_violation = is_valid(positions, inner_angles, oc, side_len, rot_outer)

        penalty = 0.0
        if not is_disjoint:
            penalty += max(0.0, -max_overlap) * 10.0
        if not is_inside:
            penalty += max(0.0, max_violation) * 10.0

        return side_len + penalty

    # ---- Optimize each shape ----
    best_result = None  # (side, inner_positions, inner_angles, oc, oa)

    for name, shape_pts in shapes:
        if elapsed() > DEADLINE - 10:
            break

        # Multi-start Nelder-Mead
        best_s_for_shape = float('inf')
        best_params_for_shape = None

        # Generate initial guesses
        starts = []
        for rot_c_deg in range(0, 60, 10):
            for rot_i_deg in range(0, 60, 10):
                cx0, cy0 = 0.0, 0.0
                rot_o_deg = 0
                starts.append([
                    math.radians(rot_c_deg),
                    math.radians(rot_i_deg),
                    cx0,
                    cy0,
                    math.radians(rot_o_deg)
                ])

        # Add a few random starts
        np.random.seed(42)
        for _ in range(5):
            starts.append([
                np.random.uniform(0, 2*math.pi),
                np.random.uniform(0, 2*math.pi),
                np.random.uniform(-5, 5),
                np.random.uniform(-5, 5),
                np.random.uniform(0, 2*math.pi)
            ])

        for start in starts:
            if elapsed() > DEADLINE - 5:
                break
            try:
                res = minimize(objective, start, args=(shape_pts,), method='Nelder-Mead',
                               options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
                if res.fun < best_s_for_shape:
                    best_s_for_shape = res.fun
                    best_params_for_shape = res.x
            except Exception:
                pass

        if best_params_for_shape is not None:
            rot_cluster, rot_inner, cx, cy, rot_outer = best_params_for_shape
            R_cluster = _rot_matrix(rot_cluster)
            positions = shape_pts @ R_cluster.T
            inner_angles = np.full(12, rot_inner)
            oc = (cx, cy)
            oa = rot_outer

            # Recompute exact side length
            all_verts = np.vstack([
                _hex_vertices(x, y, 1.0, a) for x, y, a in zip(positions[:, 0], positions[:, 1], inner_angles)
            ])
            R_outer = _rot_matrix(rot_outer)
            normals_outer = _EDGE_NORMALS @ R_outer.T
            rel_verts = all_verts - np.array(oc)
            projs = rel_verts @ normals_outer.T
            max_projs = projs.max(axis=0)
            side_len = (2.0 / math.sqrt(3)) * max_projs.max()

            # Verify validity
            is_disjoint, is_inside, _, _ = is_valid(positions, inner_angles, oc, side_len, rot_outer)
            if is_disjoint and is_inside:
                if best_result is None or side_len < best_result[0]:
                    # Build inner list: (x, y, angle_degrees)
                    inner_list = []
                    for i in range(12):
                        inner_list.append([
                            float(positions[i, 0]),
                            float(positions[i, 1]),
                            float(math.degrees(rot_inner))
                        ])
                    best_result = (side_len, inner_list, list(oc), rot_outer)

    # ---- Fallback if no valid result found ----
    if best_result is None:
        # Use 3x4 brick with known valid side
        pts = make_shape_3x4_brick()
        inner_list = [[float(x), float(y), 0.0] for x, y in pts]
        oc = [0.0, 0.0]
        oa = 0.0
        # Compute side
        all_verts = np.vstack([
            _hex_vertices(x, y, 1.0, 0.0) for x, y in pts
        ])
        normals = _EDGE_NORMALS
        projs = all_verts @ normals.T
        max_projs = projs.max(axis=0)
        side_len = (2.0 / math.sqrt(3)) * max_projs.max()
        best_result = (side_len, inner_list, oc, oa)

    best_os, best_inner, best_oc, best_oa = best_result[0], best_result[1], best_result[2], best_result[3]

    # ---- Final polish: try shrinking outer side slightly ----
    if elapsed() < DEADLINE - 2:
        # Rebuild positions and angles from best_inner
        positions = np.array([[x, y] for x, y, a in best_inner])
        inner_angles = np.array([math.radians(a) for x, y, a in best_inner])
        step = 0.005
        for _ in range(50):
            if elapsed() > DEADLINE - 1:
                break
            new_os = best_os - step
            if new_os < 2.0:
                break
            is_disjoint, is_inside, _, _ = is_valid(positions, inner_angles, tuple(best_oc), new_os, best_oa)
            if is_disjoint and is_inside:
                best_os = new_os
            else:
                step *= 0.5
                if step < 1e-6:
                    break

    return best_inner, tuple(best_oc), best_os, best_oa
