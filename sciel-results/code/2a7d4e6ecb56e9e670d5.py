import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    DEADLINE = 148.0

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

    def point_in_poly(pt, poly):
        """Ray casting point-in-polygon test."""
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

    def seg_seg_dist(p1, p2, p3, p4):
        """Minimum distance between two line segments."""
        def pt_seg_dist(p, a, b):
            ab = b - a
            ab2 = np.dot(ab, ab)
            if ab2 < 1e-15:
                return np.linalg.norm(p - a)
            t = np.dot(p - a, ab) / ab2
            t = np.clip(t, 0.0, 1.0)
            proj = a + t * ab
            return np.linalg.norm(p - proj)

        d1 = pt_seg_dist(p1, p3, p4)
        d2 = pt_seg_dist(p2, p3, p4)
        d3 = pt_seg_dist(p3, p1, p2)
        d4 = pt_seg_dist(p4, p1, p2)
        return min(d1, d2, d3, d4)

    def min_dist_convex(p1, p2):
        """Minimum distance between two convex polygons. Returns 0 if they intersect."""
        for v in p1:
            if point_in_poly(v, p2):
                return 0.0
        for v in p2:
            if point_in_poly(v, p1):
                return 0.0

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

    def is_valid_config(inner_centers, inner_angles, outer_center, outer_side, outer_angle):
        """Check disjointness and containment. Returns True if valid."""
        n = len(inner_centers)
        inner_hexes = []
        for i in range(n):
            x, y = inner_centers[i]
            inner_hexes.append(hex_vertices(x, y, 1.0, inner_angles[i]))

        for i in range(n):
            for j in range(i + 1, n):
                d = min_dist_convex(inner_hexes[i], inner_hexes[j])
                if d < 1e-9:
                    return False

        outer_hex = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
        for hx in inner_hexes:
            for v in hx:
                if not point_in_poly(v, outer_hex):
                    return False
        return True

    def min_enclosing_hex_side(inner_hexes, outer_center, outer_angle):
        """Find minimum side length of outer hexagon (at given center and angle) to contain all inner hexes."""
        min_scale = 0.0
        for hx in inner_hexes:
            for v in hx:
                for k in range(6):
                    theta = np.deg2rad(outer_angle) + np.deg2rad(30 + 60 * k)
                    nx = np.cos(theta)
                    ny = np.sin(theta)
                    proj = (v[0] - outer_center[0]) * nx + (v[1] - outer_center[1]) * ny
                    if proj > 0:
                        scale = proj / (np.sqrt(3) / 2)
                        min_scale = max(min_scale, scale)
        return min_scale

    # ---- Build the 11-hexagon flower packing ----
    eps = 1e-9
    centers = [(0.0, 0.0)]  # center
    # First ring: 6 hexagons
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3) + eps
        centers.append((r * np.cos(angle), r * np.sin(angle)))
    # Second ring: 4 hexagons
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 2 * np.sqrt(3) + eps
        centers.append((r * np.cos(angle), r * np.sin(angle)))
    
    n = len(centers)  # 11
    init_inner_angles = [30.0] * n
    init_outer_center = (0.0, 0.0)
    init_outer_angle = 30.0

    # Compute initial outer side
    inner_hexes = []
    for x, y in centers:
        inner_hexes.append(hex_vertices(x, y, 1.0, 30.0))
    init_side = min_enclosing_hex_side(inner_hexes, (0.0, 0.0), 30.0) + 1e-6

    # ---- Optimization using scipy ----
    def objective(params):
        """
        params: [cx0, cy0, ca0, cx1, cy1, ca1, ..., cx10, cy10, ca10, outer_cx, outer_cy, outer_angle, outer_side]
        Minimize outer_side subject to constraints.
        We use a penalty approach: minimize outer_side + penalty for violations.
        """
        n_hex = 11
        inner_centers = []
        inner_angles = []
        for i in range(n_hex):
            cx = params[3*i]
            cy = params[3*i + 1]
            ca = params[3*i + 2]
            inner_centers.append((cx, cy))
            inner_angles.append(ca)
        
        outer_cx = params[3*n_hex]
        outer_cy = params[3*n_hex + 1]
        outer_angle = params[3*n_hex + 2]
        outer_side = params[3*n_hex + 3]
        
        # Compute actual min enclosing side for the current configuration
        inner_hexes = []
        for i in range(n_hex):
            inner_hexes.append(hex_vertices(inner_centers[i][0], inner_centers[i][1], 1.0, inner_angles[i]))
        
        actual_side = min_enclosing_hex_side(inner_hexes, (outer_cx, outer_cy), outer_angle)
        
        # Check disjointness
        penalty = 0.0
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                d = min_dist_convex(inner_hexes[i], inner_hexes[j])
                if d < 1e-6:
                    penalty += (1e-6 - d) * 1000.0
        
        # Check containment: all inner vertices must be inside outer hexagon
        outer_hex = hex_vertices(outer_cx, outer_cy, outer_side, outer_angle)
        for hx in inner_hexes:
            for v in hx:
                if not point_in_poly(v, outer_hex):
                    # Estimate how far outside
                    # Find minimum distance to outer boundary
                    min_d = float('inf')
                    for i in range(6):
                        a1 = outer_hex[i]
                        a2 = outer_hex[(i+1) % 6]
                        d = seg_seg_dist(v, a1, a2)
                        min_d = min(min_d, d)
                    penalty += min_d * 1000.0
        
        return actual_side + penalty

    def objective_simple(params):
        """
        Simpler objective: just the min enclosing side, assuming valid configuration.
        params: [cluster_rotation, outer_rotation]
        """
        cluster_rot = params[0]
        outer_rot = params[1]
        
        cr = np.deg2rad(cluster_rot)
        rotated_centers = []
        for x, y in centers:
            rx = x * np.cos(cr) - y * np.sin(cr)
            ry = x * np.sin(cr) + y * np.cos(cr)
            rotated_centers.append((rx, ry))
        
        rotated_angles = [a + cluster_rot for a in init_inner_angles]
        
        rot_hexes = []
        for i in range(n):
            rot_hexes.append(hex_vertices(rotated_centers[i][0], rotated_centers[i][1], 1.0, rotated_angles[i]))
        
        side = min_enclosing_hex_side(rot_hexes, (0.0, 0.0), outer_rot)
        return side

    best_side = init_side
    best_cluster_angle = 0.0
    best_outer_angle = 30.0
    best_centers = centers[:]
    best_inner_angles = init_inner_angles[:]

    # Coarse grid search
    coarse_steps = 20
    for i in range(coarse_steps + 1):
        if remaining() < 10.0:
            break
        cluster_rot = i * 30.0 / coarse_steps
        for j in range(coarse_steps + 1):
            if remaining() < 10.0:
                break
            outer_rot = j * 30.0 / coarse_steps
            cr = np.deg2rad(cluster_rot)
            rotated_centers = []
            for x, y in centers:
                rx = x * np.cos(cr) - y * np.sin(cr)
                ry = x * np.sin(cr) + y * np.cos(cr)
                rotated_centers.append((rx, ry))
            rotated_angles = [a + cluster_rot for a in init_inner_angles]
            rot_hexes = []
            for ii in range(n):
                rot_hexes.append(hex_vertices(rotated_centers[ii][0], rotated_centers[ii][1], 1.0, rotated_angles[ii]))
            side = min_enclosing_hex_side(rot_hexes, (0.0, 0.0), outer_rot)
            if side < best_side:
                best_side = side
                best_cluster_angle = cluster_rot
                best_outer_angle = outer_rot
                best_centers = rotated_centers
                best_inner_angles = rotated_angles

    # Use scipy to refine
    if remaining() > 30.0:
        try:
            result = minimize(objective_simple, [best_cluster_angle, best_outer_angle],
                              method='Nelder-Mead',
                              options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000, 'maxfev': 1000})
            if result.fun < best_side:
                best_cluster_angle = result.x[0]
                best_outer_angle = result.x[1]
                best_side = result.fun
                cr = np.deg2rad(best_cluster_angle)
                rotated_centers = []
                for x, y in centers:
                    rx = x * np.cos(cr) - y * np.sin(cr)
                    ry = x * np.sin(cr) + y * np.cos(cr)
                    rotated_centers.append((rx, ry))
                best_centers = rotated_centers
                best_inner_angles = [a + best_cluster_angle for a in init_inner_angles]
        except Exception:
            pass

    # Fine grid search around the best found
    if remaining() > 20.0:
        fine_steps = 50
        delta = 0.5
        c_lo = max(0.0, best_cluster_angle - delta)
        c_hi = min(30.0, best_cluster_angle + delta)
        o_lo = max(0.0, best_outer_angle - delta)
        o_hi = min(30.0, best_outer_angle + delta)
        
        for i in range(fine_steps + 1):
            if remaining() < 15.0:
                break
            cluster_rot = c_lo + (c_hi - c_lo) * i / fine_steps
            for j in range(fine_steps + 1):
                if remaining() < 15.0:
                    break
                outer_rot = o_lo + (o_hi - o_lo) * j / fine_steps
                cr = np.deg2rad(cluster_rot)
                rotated_centers = []
                for x, y in centers:
                    rx = x * np.cos(cr) - y * np.sin(cr)
                    ry = x * np.sin(cr) + y * np.cos(cr)
                    rotated_centers.append((rx, ry))
                rotated_angles = [a + cluster_rot for a in init_inner_angles]
                rot_hexes = []
                for ii in range(n):
                    rot_hexes.append(hex_vertices(rotated_centers[ii][0], rotated_centers[ii][1], 1.0, rotated_angles[ii]))
                side = min_enclosing_hex_side(rot_hexes, (0.0, 0.0), outer_rot)
                if side < best_side:
                    best_side = side
                    best_cluster_angle = cluster_rot
                    best_outer_angle = outer_rot
                    best_centers = rotated_centers
                    best_inner_angles = rotated_angles

    # Even finer search
    if remaining() > 15.0:
        finer_steps = 100
        delta = 0.05
        c_lo = max(0.0, best_cluster_angle - delta)
        c_hi = min(30.0, best_cluster_angle + delta)
        o_lo = max(0.0, best_outer_angle - delta)
        o_hi = min(30.0, best_outer_angle + delta)
        
        for i in range(finer_steps + 1):
            if remaining() < 10.0:
                break
            cluster_rot = c_lo + (c_hi - c_lo) * i / finer_steps
            for j in range(finer_steps + 1):
                if remaining() < 10.0:
                    break
                outer_rot = o_lo + (o_hi - o_lo) * j / finer_steps
                cr = np.deg2rad(cluster_rot)
                rotated_centers = []
                for x, y in centers:
                    rx = x * np.cos(cr) - y * np.sin(cr)
                    ry = x * np.sin(cr) + y * np.cos(cr)
                    rotated_centers.append((rx, ry))
                rotated_angles = [a + cluster_rot for a in init_inner_angles]
                rot_hexes = []
                for ii in range(n):
                    rot_hexes.append(hex_vertices(rotated_centers[ii][0], rotated_centers[ii][1], 1.0, rotated_angles[ii]))
                side = min_enclosing_hex_side(rot_hexes, (0.0, 0.0), outer_rot)
                if side < best_side:
                    best_side = side
                    best_cluster_angle = cluster_rot
                    best_outer_angle = outer_rot
                    best_centers = rotated_centers
                    best_inner_angles = rotated_angles

    # Use the best configuration found
    inner_centers = best_centers
    inner_angles = best_inner_angles
    outer_center = (0.0, 0.0)
    outer_side = best_side + 1e-6  # margin for safety
    outer_angle = best_outer_angle

    # Final validity check
    if is_valid_config(inner_centers, inner_angles, outer_center, outer_side, outer_angle):
        inner = [[inner_centers[i][0], inner_centers[i][1], inner_angles[i]] for i in range(n)]
        return inner, outer_center, outer_side, outer_angle
    else:
        # Fallback: use the initial configuration with a safe margin
        inner = []
        for x, y in centers:
            inner.append([x, y, 30.0])
        return inner, (0.0, 0.0), init_side, 30.0
