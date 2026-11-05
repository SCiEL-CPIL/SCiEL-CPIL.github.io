"""Rewritten program: rigid-body cluster rotation with vectorized support functions."""
import time
import math
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # --- Build the base cluster (hexagonal lattice: 1 center, 6 ring-1, 4 ring-2) ---
    # For two adjacent unit hexagons in a honeycomb with the same orientation,
    # the center-to-center distance is 2 * apothem = sqrt(3).
    # Ring-1: 6 hexagons at distance sqrt(3), angles 0,60,...,300
    # Ring-2: 4 hexagons at distance 2*sqrt(3)? No.
    # Let's use the triangular/hex lattice: basis vectors a1 = sqrt(3)*(1,0), a2 = sqrt(3)*(-0.5, sqrt(3)/2)
    # Actually for edge-to-edge hex tiling with same orientation, neighbors are at distance sqrt(3)
    # in directions 0,60,120,180,240,300.
    # Ring-2 positions: 2*a1, 2*a2, a1+a2, a1-a2, etc.
    # Let me just use the standard 1+6+4 = 11 hexagonal number arrangement.
    # Hex numbers: 1, 7, 19, ... so 11 is NOT a hex number.
    # The seed used: 1 center, 6 at sqrt(3), 4 at distance 3 (angle 30,90,150,210,270,330 but only 4).
    # That's a valid non-overlapping config. Let me verify the seed's geometry is valid and use it.

    def build_base_cluster():
        """Return list of (x, y, angle_deg) for the 11 base hexagons (unrotated)."""
        centers = []
        centers.append((0.0, 0.0))
        # Ring 1: 6 hexagons at distance sqrt(3) * scale, angles 0,60,...,300
        # Scale > 1 ensures strict disjointness (gap between hexagons)
        scale = 1.001
        d1 = math.sqrt(3.0) * scale
        for k in range(6):
            a = math.radians(k * 60.0)
            centers.append((d1 * math.cos(a), d1 * math.sin(a)))
        # Ring 2: 4 hexagons at distance 3 * scale, angles 30,90,150,210
        d2 = 3.0 * scale
        for k in range(4):
            a = math.radians(30.0 + k * 60.0)
            centers.append((d2 * math.cos(a), d2 * math.sin(a)))
        return [(x, y, 30.0) for (x, y) in centers]

    base_cluster = build_base_cluster()

    # Precompute the 6 vertices of a unit hexagon (angle 30) as offsets from center
    def hex_vertex_offsets(angle_deg):
        a = math.radians(angle_deg)
        offs = []
        for k in range(6):
            th = a + k * math.pi / 3.0
            offs.append((math.cos(th), math.sin(th)))
        return np.array(offs)  # shape (6,2)

    base_offs = hex_vertex_offsets(30.0)  # (6,2)

    # All 66 vertices of the base cluster in local (unrotated) coordinates: shape (66, 2)
    base_vertices = []
    for (cx, cy, ang) in base_cluster:
        # For the rigid-body approach, all hexagons share the same orientation offset.
        # In base_cluster all angles are 30, so offsets are base_offs.
        for (ox, oy) in base_offs:
            base_vertices.append((cx + ox, cy + oy))
    base_vertices = np.array(base_vertices)  # (66, 2)

    def rotate_points(pts, angle_deg):
        a = math.radians(angle_deg)
        c, s = math.cos(a), math.sin(a)
        R = np.array([[c, -s], [s, c]])
        return pts @ R.T

    def cluster_vertices(cluster_angle_deg):
        """Return (66,2) array of all vertices after rotating the whole cluster."""
        return rotate_points(base_vertices, cluster_angle_deg)

    def support_function(verts, normal_angles_rad):
        """
        verts: (M,2)
        normal_angles_rad: array of angles
        Returns array of max projection of verts onto each normal direction.
        """
        normals = np.column_stack([np.cos(normal_angles_rad), np.sin(normal_angles_rad)])  # (K,2)
        # projections: (M,K)
        proj = verts @ normals.T
        return proj.max(axis=0)  # (K,)

    def required_side(cluster_angle_deg, outer_angle_deg):
        """
        Compute the minimum outer hex side length for the given cluster and outer rotations.
        Outer hex centered at origin. Its 6 outward normals are at outer_angle + 30 + k*60 degrees.
        """
        verts = cluster_vertices(cluster_angle_deg)
        normal_angles = [math.radians(outer_angle_deg + 30.0 + k * 60.0) for k in range(6)]
        h = support_function(verts, np.array(normal_angles))
        apothem_factor = math.sqrt(3.0) / 2.0
        return h.max() / apothem_factor

    # --- Optimization over (cluster_angle, outer_angle) ---
    # Both have 60-degree periodicity. Search [0, 60) for each.
    # Dense grid first.
    best_side = float('inf')
    best_pair = (0.0, 0.0)

    # Coarse grid
    n_coarse = 120
    for i in range(n_coarse):
        ca = i * 60.0 / n_coarse
        for j in range(n_coarse):
            oa = j * 60.0 / n_coarse
            s = required_side(ca, oa)
            if s < best_side:
                best_side = s
                best_pair = (ca, oa)

    if time.time() > deadline - 40:
        # Not enough time; use the coarse best
        pass
    else:
        # Refine with a finer local grid around the best point
        ca0, oa0 = best_pair
        for radius, n_fine in [(6.0, 60), (2.0, 60), (0.5, 60)]:
            for i in range(n_fine):
                ca = ca0 - radius + 2 * radius * i / (n_fine - 1)
                ca = ca % 60.0
                for j in range(n_fine):
                    oa = oa0 - radius + 2 * radius * j / (n_fine - 1)
                    oa = oa % 60.0
                    s = required_side(ca, oa)
                    if s < best_side:
                        best_side = s
                        best_pair = (ca, oa)
            ca0, oa0 = best_pair
            if time.time() > deadline - 25:
                break

        # Golden-section / Nelder-Mead refinement using scipy if available
        if time.time() < deadline - 15:
            try:
                from scipy.optimize import minimize
                def obj(p):
                    return required_side(p[0] % 60.0, p[1] % 60.0)
                res = minimize(obj, np.array(best_pair), method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
                ca_r = res.x[0] % 60.0
                oa_r = res.x[1] % 60.0
                s_r = res.fun
                if s_r < best_side:
                    best_side = s_r
                    best_pair = (ca_r, oa_r)
            except Exception:
                pass

    ca_opt, oa_opt = best_pair

    # --- Build the final rotated inner positions ---
    ca = math.radians(ca_opt)
    cos_a, sin_a = math.cos(ca), math.sin(ca)
    inner = []
    for (x, y, a) in base_cluster:
        rx = x * cos_a - y * sin_a
        ry = x * sin_a + y * cos_a
        new_a = a + ca_opt
        inner.append((rx, ry, new_a))

    # Add a tiny epsilon to the side to ensure strict containment (vertices not on boundary)
    eps = 1e-9
    final_side = best_side + eps

    # --- Validation (lightweight) ---
    def hex_vertices(cx, cy, side, angle_deg):
        a = math.radians(angle_deg)
        return [(cx + side * math.cos(a + k * math.pi / 3.0),
                 cy + side * math.sin(a + k * math.pi / 3.0)) for k in range(6)]

    def point_in_convex(p, poly):
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) < -1e-10:
                return False
        return True

    def polygon_area_signed(verts):
        area = 0.0
        n = len(verts)
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return area / 2.0

    def line_intersect(p1, p2, p3, p4):
        x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-15:
            return p1
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    def overlap_area(v1, v2):
        def clip(subject, clip_poly):
            out = list(subject)
            for i in range(len(clip_poly)):
                if not out:
                    break
                inp = out
                out = []
                S = inp[-1]
                for E in inp:
                    if point_in_convex(E, clip_poly):
                        if not point_in_convex(S, clip_poly):
                            out.append(line_intersect(S, E, clip_poly[i], clip_poly[(i + 1) % len(clip_poly)]))
                        out.append(E)
                    elif point_in_convex(S, clip_poly):
                        out.append(line_intersect(S, E, clip_poly[i], clip_poly[(i + 1) % len(clip_poly)]))
                    S = E
            return out
        cl = clip(v1, v2)
        if len(cl) < 3:
            return 0.0
        return abs(polygon_area_signed(cl))

    # Check containment
    outer_verts = hex_vertices(0.0, 0.0, final_side, oa_opt)
    valid = True
    for (ix, iy, iang) in inner:
        iv = hex_vertices(ix, iy, 1.0, iang)
        for v in iv:
            if not point_in_convex(v, outer_verts):
                valid = False
                break
        if not valid:
            break

    # Check non-overlap (only if still valid and time remains)
    if valid and time.time() < deadline - 3:
        for i in range(N):
            vi = hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            for j in range(i + 1, N):
                vj = hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
                if overlap_area(vi, vj) > 1e-7:
                    valid = False
                    break
            if not valid:
                break

    if not valid:
        # Fallback: use the seed's analytic solution
        centers = [(0.0, 0.0)]
        d1 = math.sqrt(3.0)
        for k in range(6):
            a = math.radians(k * 60.0)
            centers.append((d1 * math.cos(a), d1 * math.sin(a)))
        d2 = 3.0
        for k in range(4):
            a = math.radians(30.0 + k * 60.0)
            centers.append((d2 * math.cos(a), d2 * math.sin(a)))
        inner_fb = [(x, y, 30.0) for (x, y) in centers]
        max_x = 2 * math.sqrt(3)
        max_y = 4.0
        s_x = 2 * max_x / math.sqrt(3)
        s_y = max_y
        side_fb = max(s_x, s_y) + 1e-9
        return [[p[0], p[1], p[2]] for p in inner_fb], (0.0, 0.0), side_fb, 30.0

    return [[p[0], p[1], p[2]] for p in inner], (0.0, 0.0), final_side, oa_opt
# EVOLVE-BLOCK-END
