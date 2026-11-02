"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side.

Strategy:
  - Enumerate many distinct topologies of 12 hexagons on a hexagonal lattice
    (brick 3x4, 4x3, 2x6, 6x2, and various hexagonal clusters).
  - For each topology, compute the convex hull of all 72 vertices.
  - Optimize the global rotation of the rigid cluster and the enclosing
    hexagon's orientation jointly, using a fast convex-hull-based method.
  - Use scipy's minimize for local refinement of rotation and center.
  - Verify validity with exact SAT checks.
"""

import math
import time
import numpy as np
from itertools import product

N = 12
DEADLINE = 135.0  # safety margin before 150s wall-clock kill

def construct():
    t0 = time.time()

    def elapsed():
        return time.time() - t0

    # ---- Geometry helpers ----

    def hex_vertices(cx, cy, side, angle_rad):
        """Return 6x2 array of vertices for a regular hexagon."""
        angs = angle_rad + np.array([0.0, math.pi/3, 2*math.pi/3,
                                      math.pi, 4*math.pi/3, 5*math.pi/3])
        return np.column_stack([cx + side * np.cos(angs),
                                cy + side * np.sin(angs)])

    def rot_matrix(angle_rad):
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        return np.array([[c, -s], [s, c]], dtype=float)

    def convex_hull(points):
        """Andrew's monotone chain convex hull. Returns hull vertices CCW."""
        points = np.array(points)
        points = points[np.lexsort((points[:, 1], points[:, 0]))]
        if len(points) <= 1:
            return points
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        lower = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in points[::-1]:
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        return np.vstack([lower[:-1], upper[:-1]])

    # ---- Fast validity check using SAT ----

    def sat_disjoint(v1, v2, tol=1e-9):
        """Check if two convex polygons (6x2 each) are disjoint.
        Returns True if disjoint (no overlap)."""
        for i in range(6):
            e = v1[(i + 1) % 6] - v1[i]
            n = np.array([-e[1], e[0]])
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n /= nn
            p1 = v1 @ n
            p2 = v2 @ n
            # If intervals don't overlap, polygons are disjoint
            if p1.max() < p2.min() - tol or p2.max() < p1.min() - tol:
                return True
        for i in range(6):
            e = v2[(i + 1) % 6] - v2[i]
            n = np.array([-e[1], e[0]])
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n /= nn
            p1 = v1 @ n
            p2 = v2 @ n
            if p1.max() < p2.min() - tol or p2.max() < p1.min() - tol:
                return True
        return False

    def point_in_hex(px, py, cx, cy, side, angle_rad, tol=1e-9):
        """Check if point is inside hexagon."""
        dx, dy = px - cx, py - cy
        c, s = math.cos(-angle_rad), math.sin(-angle_rad)
        lx = dx * c - dy * s
        ly = dx * s + dy * c
        apothem = side * math.sqrt(3) / 2
        for k in range(6):
            na = math.radians(30 + 60 * k)
            if lx * math.cos(na) + ly * math.sin(na) > apothem + tol:
                return False
        return True

    def all_inside(inner, oc, os, oa, tol=1e-9):
        """Check all inner hexagons are inside outer hexagon."""
        for x, y, a in inner:
            vs = hex_vertices(x, y, 1.0, a)
            for vx, vy in vs:
                if not point_in_hex(vx, vy, oc[0], oc[1], os, oa, tol):
                    return False
        return True

    def pairwise_disjoint(inner, tol=1e-9):
        """Check all pairs of inner hexagons are disjoint."""
        verts = [hex_vertices(x, y, 1.0, a) for x, y, a in inner]
        for i in range(len(inner)):
            for j in range(i + 1, len(inner)):
                if not sat_disjoint(verts[i], verts[j], tol):
                    return False
        return True

    def is_valid(inner, oc, os, oa):
        return all_inside(inner, oc, os, oa) and pairwise_disjoint(inner)

    # ---- Minimum enclosing hexagon for a set of points ----

    def enclosing_hex_side(points, angle_rad):
        """Compute minimal side length and center of a regular hexagon
        (rotated by angle_rad) enclosing all points.
        Returns (side, center_tuple)."""
        # 6 outward normals for the enclosing hexagon
        angs = angle_rad + np.array([math.pi/6 + k * math.pi/3 for k in range(6)])
        normals = np.column_stack([np.cos(angs), np.sin(angs)])  # 6x2
        P = np.array(points)  # Mx2

        def required_side(c):
            projs = P @ normals.T  # Mx6
            maxproj = projs.max(axis=0)  # 6
            vals = maxproj - normals @ c  # 6
            return (2.0 / math.sqrt(3)) * vals.max()

        # Grid search for center
        xs = P[:, 0]
        ys = P[:, 1]
        cx0, cx1 = xs.min(), xs.max()
        cy0, cy1 = ys.min(), ys.max()
        best_s = float('inf')
        best_c = ((cx0 + cx1) / 2, (cy0 + cy1) / 2)
        for gi in range(15):
            cx = cx0 + (cx1 - cx0) * gi / 14
            for gj in range(15):
                cy = cy0 + (cy1 - cy0) * gj / 14
                s = required_side((cx, cy))
                if s < best_s:
                    best_s = s
                    best_c = (cx, cy)
        # Local refinement
        try:
            from scipy.optimize import minimize
            def obj(c):
                return required_side(c)
            res = minimize(obj, best_c, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-9})
            if res.fun < best_s:
                best_s = res.fun
                best_c = (res.x[0], res.x[1])
        except Exception:
            pass
        return best_s, best_c

    # ---- Generate candidate topologies on hexagonal lattice ----

    def make_3x4_brick():
        """3 rows x 4 cols, pointy-top brick pattern."""
        pts = []
        for r in range(3):
            for c in range(4):
                x = (c - 1.5) * math.sqrt(3) + (math.sqrt(3) / 2 if r % 2 else 0)
                y = (r - 1.0) * 1.5
                pts.append((x, y))
        return pts

    def make_4x3_brick():
        """4 rows x 3 cols, pointy-top brick pattern."""
        pts = []
        for r in range(4):
            for c in range(3):
                x = (c - 1.0) * math.sqrt(3) + (math.sqrt(3) / 2 if r % 2 else 0)
                y = (r - 1.5) * 1.5
                pts.append((x, y))
        return pts

    def make_2x6_flat():
        """2 rows x 6 cols, flat-top."""
        pts = []
        for r in range(2):
            for c in range(6):
                x = (c - 2.5) * 3.0
                y = (r - 0.5) * math.sqrt(3)
                pts.append((x, y))
        return pts

    def make_6x2_flat():
        """6 rows x 2 cols, flat-top."""
        pts = []
        for r in range(6):
            for c in range(2):
                x = (c - 0.5) * 3.0
                y = (r - 2.5) * math.sqrt(3)
                pts.append((x, y))
        return pts

    def make_hex_cluster_12():
        """Center + ring1(6) + 5 from ring2 = 12, hexagonal close-packed."""
        pts = [(0, 0)]
        # Ring 1: 6 hexes at distance sqrt(3)
        for k in range(6):
            a = math.pi / 3 * k
            pts.append((math.sqrt(3) * math.cos(a), math.sqrt(3) * math.sin(a)))
        # Ring 2: pick 5 contiguous positions
        # Ring 2 positions are at various distances
        # For pointy-top hex lattice, ring 2 has 12 positions
        # Let's add 5 more to make 12 total
        # Positions at distance 2*sqrt(3) in certain directions
        for k in range(5):
            a = math.pi / 3 * k + math.pi / 6
            pts.append((2 * math.sqrt(3) * math.cos(a), 2 * math.sqrt(3) * math.sin(a)))
        return pts

    def make_3x4_compact():
        """Alternative 3x4 with tighter spacing."""
        pts = []
        for r in range(3):
            for c in range(4):
                x = (c - 1.5) * math.sqrt(3)
                y = (r - 1.0) * 1.5 + (math.sqrt(3) / 2 if r % 2 else 0)
                pts.append((x, y))
        return pts

    def make_staggered_2x6():
        """2 rows x 6 cols with vertical stagger."""
        pts = []
        for c in range(6):
            x = (c - 2.5) * 3.0
            pts.append((x, -math.sqrt(3) / 2))
            pts.append((x, math.sqrt(3) / 2))
        return pts

    def make_4x3_compact():
        """Alternative 4x3 with tighter spacing."""
        pts = []
        for r in range(4):
            for c in range(3):
                x = (c - 1.0) * math.sqrt(3)
                y = (r - 1.5) * 1.5 + (math.sqrt(3) / 2 if r % 2 else 0)
                pts.append((x, y))
        return pts

    # Collect all valid candidates
    candidates = []
    for name, pts in [
        ("3x4brick", make_3x4_brick()),
        ("4x3brick", make_4x3_brick()),
        ("2x6flat", make_2x6_flat()),
        ("6x2flat", make_6x2_flat()),
        ("hex12", make_hex_cluster_12()),
        ("3x4compact", make_3x4_compact()),
        ("2x6stagger", make_staggered_2x6()),
        ("4x3compact", make_4x3_compact()),
    ]:
        if pts and len(pts) == 12:
            candidates.append((name, pts))

    # ---- Evaluate each candidate ----

    best_result = None  # (side, inner, oc, os, oa)

    for name, pts in candidates:
        if elapsed() > DEADLINE - 5:
            break

        best_s_for_cluster = float('inf')
        best_rot = 0.0
        best_center = (0, 0)

        # Coarse scan of global rotation (0 to 60 degrees, period of hex symmetry)
        for deg in range(0, 60, 1):
            rot = math.radians(deg)
            R = rot_matrix(rot)
            P = np.array(pts) @ R.T
            s, c = enclosing_hex_side(P.tolist(), rot)
            if s < best_s_for_cluster:
                best_s_for_cluster = s
                best_rot = rot
                best_center = c

        # Fine scan around best
        for deg100 in range(int(best_rot * 10000) - 200, int(best_rot * 10000) + 200, 5):
            rot = math.radians(deg100 / 10000.0)
            R = rot_matrix(rot)
            P = np.array(pts) @ R.T
            s, c = enclosing_hex_side(P.tolist(), rot)
            if s < best_s_for_cluster:
                best_s_for_cluster = s
                best_rot = rot
                best_center = c

        # Build inner list
        inner = []
        R = rot_matrix(best_rot)
        for px, py in pts:
            rx, ry = R @ np.array([px, py])
            inner.append([rx + best_center[0], ry + best_center[1], best_rot])

        # Verify validity
        if is_valid(inner, best_center, best_s_for_cluster, best_rot):
            if best_result is None or best_s_for_cluster < best_result[0]:
                best_result = (best_s_for_cluster, inner, list(best_center), best_s_for_cluster, best_rot)

    # ---- Fallback ----
    if best_result is None:
        inner = make_2x6_flat()
        best_result = (9.0, inner, [0.0, 0.0], 9.0, 0.0)

    best_os, best_inner, best_oc, best_oa = best_result[0], best_result[1], best_result[2], best_result[4]

    # ---- Local polish: try shrinking os slightly ----
    if elapsed() < DEADLINE - 2:
        step = 0.005
        for _ in range(150):
            if elapsed() > DEADLINE - 1:
                break
            new_os = best_os - step
            if new_os < 2.0:
                break
            if is_valid(best_inner, best_oc, new_os, best_oa):
                best_os = new_os
            else:
                step *= 0.5
                if step < 1e-7:
                    break

    # ---- Try small rotation tweaks to reduce enclosing side ----
    if elapsed() < DEADLINE - 3:
        for ddeg in [0.5, -0.5, 1.0, -1.0, 0.25, -0.25, 0.1, -0.1]:
            if elapsed() > DEADLINE - 2:
                break
            new_oa = best_oa + math.radians(ddeg)
            # Re-enclose with new rotation
            R = rot_matrix(new_oa - best_oa)
            new_inner = []
            for x, y, a in best_inner:
                dx, dy = x - best_oc[0], y - best_oc[1]
                nx, ny = R @ np.array([dx, dy])
                new_inner.append([nx + best_oc[0], ny + best_oc[1], a + math.radians(ddeg)])
            # Recompute enclosing side
            pts = [(x, y) for x, y, a in new_inner]
            s, c = enclosing_hex_side(pts, new_oa)
            if s < best_os - 1e-6 and is_valid(new_inner, c, s, new_oa):
                best_inner = new_inner
                best_oc = list(c)
                best_os = s
                best_oa = new_oa

    return best_inner, tuple(best_oc), best_os, best_oa
