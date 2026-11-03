"""Rewritten program: pack 11 unit hexagons into minimal regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    # Unit hexagon vertices (pointy-top, vertex at angle 0)
    base_verts = np.array([(math.cos(math.radians(a)), math.sin(math.radians(a)))
                           for a in range(0, 360, 60)])

    def hex_verts(cx, cy, angle_deg):
        """Return 6 vertices of a unit hexagon centered at (cx,cy) rotated by angle_deg."""
        a = math.radians(angle_deg)
        R = np.array([[math.cos(a), -math.sin(a)],
                      [math.sin(a), math.cos(a)]])
        return cx + R @ base_verts.T  # shape (2, 6)

    def min_outer_side(points, ocx, ocy, oangle_deg):
        """
        Minimum side length S of a regular hexagon (center ocx,ocy, rotation oangle_deg)
        containing all points. points: list/array of (x,y).
        A regular hexagon with side S has apothem = S*sqrt(3)/2.
        The 6 outward normals are at angles oangle + 30 + 60*i degrees.
        For each normal direction, the max projection of points from center must be <= apothem.
        """
        theta = math.radians(oangle_deg)
        max_proj = 0.0
        for i in range(6):
            na = theta + math.pi/6 + i * math.pi/3
            nx, ny = math.cos(na), math.sin(na)
            # Project all points
            dx = np.array([p[0] - ocx for p in points])
            dy = np.array([p[1] - ocy for p in points])
            proj = dx * nx + dy * ny
            mp = np.max(proj)
            if mp > max_proj:
                max_proj = mp
        return max_proj * 2.0 / math.sqrt(3)

    def point_in_hex(px, py, ocx, ocy, side, oangle_deg):
        """Check if point (px,py) is inside regular hexagon."""
        theta = math.radians(oangle_deg)
        apothem = side * math.sqrt(3) / 2
        for i in range(6):
            na = theta + math.pi/6 + i * math.pi/3
            nx, ny = math.cos(na), math.sin(na)
            proj = (px - ocx) * nx + (py - ocy) * ny
            if proj > apothem + 1e-9:
                return False
        return True

    def seg_intersect(p1, p2, p3, p4):
        """Check if segment p1-p2 intersects segment p3-p4."""
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False

    def hexagons_disjoint(v1, v2):
        """
        Check if two hexagons (each as array shape (2,6)) are disjoint (no overlap).
        Returns True if disjoint (valid), False if overlapping.
        """
        # Check if any vertex of one is inside the other
        for i in range(6):
            px, py = v1[0, i], v1[1, i]
            if point_in_convex(px, py, v2):
                return False
        for i in range(6):
            px, py = v2[0, i], v2[1, i]
            if point_in_convex(px, py, v1):
                return False
        # Check edge intersections
        for i in range(6):
            j = (i + 1) % 6
            for k in range(6):
                l = (k + 1) % 6
                if seg_intersect((v1[0,i], v1[1,i]), (v1[0,j], v1[1,j]),
                                  (v2[0,k], v2[1,k]), (v2[0,l], v2[1,l])):
                    return False
        return True

    def point_in_convex(px, py, verts):
        """Check if point is inside convex polygon (verts shape (2,6), CCW)."""
        # Use cross product sign consistency
        sign = 0
        for i in range(6):
            j = (i + 1) % 6
            x1, y1 = verts[0, i], verts[1, i]
            x2, y2 = verts[0, j], verts[1, j]
            c = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
            if c > 1e-10:
                if sign < 0:
                    return False
                sign = 1
            elif c < -1e-10:
                if sign > 0:
                    return False
                sign = -1
        return True

    def compute_all_vertices(inner):
        """inner: list of [x, y, angle_deg]. Returns list of (2,6) arrays."""
        verts = []
        for hx, hy, ha in inner:
            verts.append(hex_verts(hx, hy, ha))
        return verts

    def check_valid(inner, ocx, ocy, side, oangle):
        """Full validity check: no overlaps, all inside outer."""
        verts = compute_all_vertices(inner)
        # Overlap check
        for i in range(N):
            for j in range(i + 1, N):
                if not hexagons_disjoint(verts[i], verts[j]):
                    return False
        # Containment check
        for i in range(N):
            for k in range(6):
                px, py = verts[i][0, k], verts[i][1, k]
                if not point_in_hex(px, py, ocx, ocy, side, oangle):
                    return False
        return True

    def objective(params):
        """
        params: [x1,y1,a1, x2,y2,a2, ..., x11,y11,a11, ocx, ocy, oangle]
        Returns outer side length (to minimize).
        """
        inner = []
        for i in range(N):
            x, y, a = params[3*i], params[3*i+1], params[3*i+2]
            inner.append([x, y, a])
        ocx = params[3*N]
        ocy = params[3*N+1]
        oangle = params[3*N+2]

        # Quick overlap check
        verts = compute_all_vertices(inner)
        for i in range(N):
            for j in range(i + 1, N):
                if not hexagons_disjoint(verts[i], verts[j]):
                    return 1e6  # penalty

        # All vertices
        all_pts = []
        for v in verts:
            for k in range(6):
                all_pts.append((v[0, k], v[1, k]))

        side = min_outer_side(all_pts, ocx, ocy, oangle)

        # Containment check
        for i in range(N):
            for k in range(6):
                px, py = verts[i][0, k], verts[i][1, k]
                if not point_in_hex(px, py, ocx, ocy, side, oangle):
                    return 1e6

        return side

    def make_hexagonal_seed():
        """Honeycomb packing of 11 hexagons."""
        s3 = math.sqrt(3)
        inner = [[0.0, 0.0, 0.0]]
        # Ring 1: 6 hexagons at distance s3
        for i in range(6):
            a = math.radians(60 * i)
            inner.append([s3 * math.cos(a), s3 * math.sin(a), 0.0])
        # Ring 2: 4 more hexagons (to get 11 total)
        # Place at alternating ring-2 positions
        for i in [0, 2, 4, 6]:
            a = math.radians(60 * i)
            inner.append([2 * s3 * math.cos(a), 2 * s3 * math.sin(a), 0.0])
        return inner[:11]

    def make_compact_seed():
        """A more compact initial arrangement."""
        s3 = math.sqrt(3)
        inner = [[0.0, 0.0, 0.0]]
        for i in range(6):
            a = math.radians(60 * i)
            inner.append([s3 * math.cos(a), s3 * math.sin(a), 0.0])
        # Add 4 more in a tighter pattern
        # Place at 120-degree intervals at distance ~1.8
        for i in range(4):
            a = math.radians(120 * i + 30)
            inner.append([1.8 * math.cos(a), 1.8 * math.sin(a), 0.0])
        return inner[:11]

    def make_grid_seed():
        """Rectangular-ish grid."""
        s3 = math.sqrt(3)
        inner = []
        positions = [
            (0, 0), (2, 0), (4, 0),
            (1, s3), (3, s3), (5, s3),
            (0, 2*s3), (2, 2*s3), (4, 2*s3),
            (1, 3*s3), (3, 3*s3),
        ]
        for x, y in positions[:11]:
            inner.append([x, y, 0.0])
        # Center
        cx = sum(p[0] for p in inner) / len(inner)
        cy = sum(p[1] for p in inner) / len(inner)
        inner = [[p[0] - cx, p[1] - cy, p[2]] for p in inner]
        return inner

    # Generate initial seeds
    seeds = [make_hexagonal_seed(), make_compact_seed(), make_grid_seed()]

    best_side = float('inf')
    best_params = None

    for seed_idx, inner in enumerate(seeds):
        if time.time() > deadline - 20:
            break

        # Compute initial outer hexagon
        all_pts = []
        for hx, hy, ha in inner:
            v = hex_verts(hx, hy, ha)
            for k in range(6):
                all_pts.append((v[0, k], v[1, k]))

        # Try different outer angles
        best_seed_side = float('inf')
        best_seed_params = None

        for oangle in [0, 30, 60, 15, 45]:
            if time.time() > deadline - 15:
                break

            # Optimize center
            best_cx, best_cy = 0.0, 0.0
            best_c_side = float('inf')
            for dx in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                for dy in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                    s = min_outer_side(all_pts, dx, dy, oangle)
                    if s < best_c_side:
                        best_c_side = s
                        best_cx, best_cy = dx, dy

            # Build params
            params = []
            for hx, hy, ha in inner:
                params.extend([hx, hy, ha])
            params.extend([best_cx, best_cy, oangle])

            # Validate
            test_inner = [[params[3*i], params[3*i+1], params[3*i+2]] for i in range(N)]
            if check_valid(test_inner, best_cx, best_cy, best_c_side + 1e-7, oangle):
                if best_c_side < best_seed_side:
                    best_seed_side = best_c_side
                    best_seed_params = params[:]

        if best_seed_params is not None:
            if best_seed_side < best_side:
                best_side = best_seed_side
                best_params = best_seed_params[:]

        # Refine with Nelder-Mead
        if best_seed_params is not None and time.time() < deadline - 10:
            try:
                result = minimize(
                    objective,
                    best_seed_params,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 300,
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                if result.fun < 1e5 and result.fun < best_side:
                    # Validate the result
                    rp = result.x
                    r_inner = [[rp[3*i], rp[3*i+1], rp[3*i+2]] for i in range(N)]
                    roc_x, roc_y, roa = rp[3*N], rp[3*N+1], rp[3*N+2]
                    if check_valid(r_inner, roc_x, roc_y, result.fun + 1e-7, roa):
                        best_side = result.fun
                        best_params = rp.tolist()
            except Exception:
                pass

    # Final refinement with more iterations if time permits
    if best_params is not None and time.time() < deadline - 5:
        try:
            result = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-7,
                    'fatol': 1e-9,
                    'adaptive': True
                }
            )
            if result.fun < 1e5 and result.fun < best_side:
                rp = result.x
                r_inner = [[rp[3*i], rp[3*i+1], rp[3*i+2]] for i in range(N)]
                roc_x, roc_y, roa = rp[3*N], rp[3*N+1], rp[3*N+2]
                if check_valid(r_inner, roc_x, roc_y, result.fun + 1e-7, roa):
                    best_side = result.fun
                    best_params = rp.tolist()
        except Exception:
            pass

    # Build final result
    if best_params is None:
        # Fallback: hexagonal seed with generous outer hexagon
        inner = make_hexagonal_seed()
        all_pts = []
        for hx, hy, ha in inner:
            v = hex_verts(hx, hy, ha)
            for k in range(6):
                all_pts.append((v[0, k], v[1, k]))
        side = min_outer_side(all_pts, 0.0, 0.0, 0.0) + 0.5
        return inner, (0.0, 0.0), side, 0.0

    # Extract from best_params
    inner = [[best_params[3*i], best_params[3*i+1], best_params[3*i+2]] for i in range(N)]
    ocx = best_params[3*N]
    ocy = best_params[3*N+1]
    oangle = best_params[3*N+2]

    # Recompute exact side for safety
    all_pts = []
    for hx, hy, ha in inner:
        v = hex_verts(hx, hy, ha)
        for k in range(6):
            all_pts.append((v[0, k], v[1, k]))
    side = min_outer_side(all_pts, ocx, ocy, oangle) + 1e-7

    return inner, (ocx, ocy), side, oangle

# EVOLVE-BLOCK-END
