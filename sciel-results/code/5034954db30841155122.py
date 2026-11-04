import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    DEADLINE = 140.0

    def elapsed():
        return time.time() - start_time

    def remaining():
        return DEADLINE - elapsed()

    def hex_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of a regular hexagon (center cx,cy, side length, rotation in degrees)."""
        a = np.deg2rad(angle_deg)
        verts = []
        for k in range(6):
            theta = a + np.deg2rad(60 * k)
            x = cx + side * np.cos(theta)
            y = cy + side * np.sin(theta)
            verts.append([x, y])
        return np.array(verts)

    def min_dist_convex(p1, p2):
        """Minimum distance between two convex polygons (numpy arrays of vertices).
        Returns 0 if they intersect (including touching)."""
        # Check if any vertex of p1 is inside p2 or vice versa
        def point_in_poly(pt, poly):
            # Ray casting
            n = len(poly)
            inside = False
            j = n - 1
            for i in range(n):
                xi, yi = poly[i]
                xj, yj = poly[j]
                if ((yi > pt[1]) != (yj > pt[1])) and \
                   (pt[0] < (xj - xi) * (pt[1] - yi) / (yj - yi) + xi):
                    inside = not inside
                j = i
            return inside

        # Check vertex-in-polygon
        for v in p1:
            if point_in_poly(v, p2):
                return 0.0
        for v in p2:
            if point_in_poly(v, p1):
                return 0.0

        # Compute min distance between all edge pairs
        min_d = float('inf')
        for i in range(len(p1)):
            a1 = p1[i]
            a2 = p1[(i + 1) % len(p1)]
            for j in range(len(p2)):
                b1 = p2[j]
                b2 = p2[(j + 1) % len(p2)]
                d = seg_seg_dist(a1, a2, b1, b2)
                if d < min_d:
                    min_d = d
        return min_d

    def seg_seg_dist(p1, p2, p3, p4):
        """Minimum distance between two line segments."""
        def pt_seg_dist(p, a, b):
            ab = b - a
            t = np.dot(p - a, ab) / np.dot(ab, ab)
            t = np.clip(t, 0.0, 1.0)
            proj = a + t * ab
            return np.linalg.norm(p - proj)

        d1 = pt_seg_dist(p1, p3, p4)
        d2 = pt_seg_dist(p2, p3, p4)
        d3 = pt_seg_dist(p3, p1, p2)
        d4 = pt_seg_dist(p4, p1, p2)
        return min(d1, d2, d3, d4)

    def is_valid_config(inner_centers, inner_angles, outer_center, outer_side, outer_angle):
        """Check disjointness and containment. Returns True if valid."""
        n = len(inner_centers)
        inner_hexes = []
        for i in range(n):
            x, y = inner_centers[i]
            inner_hexes.append(hex_vertices(x, y, 1.0, inner_angles[i]))

        # Check pairwise disjointness (interiors must not overlap)
        for i in range(n):
            for j in range(i + 1, n):
                d = min_dist_convex(inner_hexes[i], inner_hexes[j])
                if d < 1e-9:
                    return False

        # Check containment: each inner vertex inside outer hexagon
        outer_hex = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
        for hx in inner_hexes:
            for v in hx:
                if not point_in_poly(v, outer_hex):
                    return False
        return True

    def point_in_poly(pt, poly):
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > pt[1]) != (yj > pt[1])) and \
               (pt[0] < (xj - xi) * (pt[1] - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    # ---- Initial configuration: concentric ring packing (same as current) ----
    eps = 1e-7
    centers = [(0.0, 0.0)]
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3) + eps
        centers.append((r * np.cos(angle), r * np.sin(angle)))
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0 + 2 * eps
        centers.append((r * np.cos(angle), r * np.sin(angle)))

    n = len(centers)
    init_inner_angles = [30.0] * n
    init_outer_center = (0.0, 0.0)
    init_outer_angle = 30.0

    # Compute initial outer side
    max_x = 0.0
    max_y = 0.0
    for x, y in centers:
        max_x = max(max_x, x + np.sqrt(3) / 2, -x + np.sqrt(3) / 2)
        max_y = max(max_y, y + 1.0, -y + 1.0)
    s_x = 2 * max_x / np.sqrt(3)
    s_y = max_y
    init_side = max(s_x, s_y) + 1e-6

    # ---- Build initial parameter vector ----
    # [x0,y0,th0, x1,y1,th1, ..., xn-1,yn-1,thn-1, cx, cy, S, phi]
    x0 = []
    for i in range(n):
        x0.append(centers[i][0])
        x0.append(centers[i][1])
        x0.append(np.deg2rad(init_inner_angles[i]))
    x0.append(init_outer_center[0])
    x0.append(init_outer_center[1])
    x0.append(init_side)
    x0.append(np.deg2rad(init_outer_angle))
    x0 = np.array(x0)

    def objective(params):
        if remaining() < 5.0:
            return 1e10
        # Unpack
        inner_centers = []
        inner_angles = []
        for i in range(n):
            inner_centers.append((params[3*i], params[3*i+1]))
            inner_angles.append(np.rad2deg(params[3*i+2]))
        cx = params[3*n]
        cy = params[3*n+1]
        S = params[3*n+2]
        phi = np.rad2deg(params[3*n+3])

        if S < 0.5:
            return 1e10

        # Compute inner hexes
        inner_hexes = []
        for i in range(n):
            inner_hexes.append(hex_vertices(inner_centers[i][0], inner_centers[i][1], 1.0, inner_angles[i]))

        # Overlap penalty: for each pair, penalty if distance < gap
        gap = 1e-6
        overlap_pen = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = min_dist_convex(inner_hexes[i], inner_hexes[j])
                if d < gap:
                    overlap_pen += (gap - d) ** 2 * 1000.0

        # Containment penalty: each inner vertex must be inside outer hex
        outer_hex = hex_vertices(cx, cy, S, phi)
        contain_pen = 0.0
        for hx in inner_hexes:
            for v in hx:
                if not point_in_poly(v, outer_hex):
                    # Distance to boundary (approximate)
                    contain_pen += 100.0

        return S + overlap_pen + contain_pen

    # Try to optimize
    best_params = x0.copy()
    best_obj = objective(x0)

    if remaining() > 30.0:
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8,
                                    'adaptive': True})
            if remaining() > 5.0:
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_params = res.x
        except Exception:
            pass

    # Extract best solution
    inner_centers = []
    inner_angles = []
    for i in range(n):
        inner_centers.append((best_params[3*i], best_params[3*i+1]))
        inner_angles.append(np.rad2deg(best_params[3*i+2]))
    cx = best_params[3*n]
    cy = best_params[3*n+1]
    S = best_params[3*n+2]
    phi = np.rad2deg(best_params[3*n+3])

    # Final validity check and fallback
    if is_valid_config(inner_centers, inner_angles, (cx, cy), S, phi):
        inner = [[inner_centers[i][0], inner_centers[i][1], inner_angles[i]] for i in range(n)]
        return inner, (cx, cy), S, phi
    else:
        # Fallback to original static solution
        inner = []
        for x, y in centers:
            inner.append([x, y, 30.0])
        return inner, (0.0, 0.0), init_side, 30.0
