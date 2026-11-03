"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint, linprog
from scipy.spatial import ConvexHull

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill

    # Unit hexagon vertices (flat-top, side 1)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    hex_verts = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(angle_deg):
        """Get rotated hexagon vertices."""
        theta = math.radians(angle_deg)
        R = np.array([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta), math.cos(theta)]])
        return hex_verts @ R.T

    def hexagon_support(x, y, angle_deg, direction):
        """Support function of hexagon at (x,y) with rotation angle_deg in direction (dx,dy)."""
        verts = get_hex_verts(angle_deg)
        pts = verts + np.array([x, y])
        return np.max(pts @ direction)

    def is_hexagon_inside_outer(x, y, angle_deg, outer_center, outer_side, outer_angle):
        """Check if inner hexagon at (x,y,angle_deg) is inside outer hexagon."""
        # Outer hexagon: center, side length s, rotation
        # A point is inside a regular hexagon if it satisfies 6 half-space constraints
        # The hexagon has vertices at distance s from center (circumradius = s)
        # Inradius = s * sqrt(3)/2
        # The 6 normal directions are at angles outer_angle + 30 + k*60
        ok = True
        for k in range(6):
            normal_angle = math.radians(outer_angle + 30 + k * 60)
            normal = np.array([math.cos(normal_angle), math.sin(normal_angle)])
            # The boundary is at distance inradius = s * sqrt(3)/2 from center
            inradius = outer_side * math.sqrt(3) / 2
            # Check all vertices of inner hexagon
            inner_verts = get_hex_verts(angle_deg) + np.array([x, y])
            for v in inner_verts:
                val = (v - np.array(outer_center)) @ normal
                if val > inradius + 1e-9:
                    ok = False
                    break
            if not ok:
                break
        return ok

    def check_non_overlap(inner_hexes, tol=1e-7):
        """Check if any two inner hexagons overlap. Returns True if no overlap."""
        n = len(inner_hexes)
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi, ai = inner_hexes[i]
                xj, yj, aj = inner_hexes[j]
                # Check if any vertex of i is inside j or vice versa, or edges intersect
                # Simple check: distance between centers vs minimum distance for non-overlap
                # For two unit hexagons, the minimum center distance for non-overlap depends on relative rotation
                # A conservative check: if distance < sqrt(3) - epsilon, they might overlap
                # But this isn't tight. Let's do a proper check using support functions or vertex containment
                # Quick reject: if distance > 2 * sqrt(3), definitely no overlap (max diameter is 2*sqrt(3)/2 * 2 = 2*sqrt(3)? No)
                # Max distance between two points in unit hexagon is 2 (diameter through opposite vertices = 2)
                # So if center distance > 2 + 2 = 4, definitely no overlap. But we need tighter.
                # Let's check if any vertex of one is inside the other
                verts_i = get_hex_verts(ai) + np.array([xi, yi])
                verts_j = get_hex_verts(aj) + np.array([xj, yj])
                
                # Check if any vertex of i is inside j
                # A point is inside a convex polygon if it's on the same side of all edges
                def point_in_convex_poly(pt, poly):
                    n_poly = len(poly)
                    sign = 0
                    for k in range(n_poly):
                        p1 = poly[k]
                        p2 = poly[(k + 1) % n_poly]
                        cross = (p2[0] - p1[0]) * (pt[1] - p1[1]) - (p2[1] - p1[1]) * (pt[0] - p1[0])
                        if cross > 1e-9:
                            if sign < 0:
                                return False
                            sign = 1
                        elif cross < -1e-9:
                            if sign > 0:
                                return False
                            sign = -1
                    return True
                
                for v in verts_i:
                    if point_in_convex_poly(v, verts_j):
                        return False
                for v in verts_j:
                    if point_in_convex_poly(v, verts_i):
                        return False
                # Also check edge intersections for robustness
                # For now, vertex containment is a good proxy for non-overlap of convex shapes
                # Actually, two convex shapes can overlap without vertex containment (edge-edge intersection)
                # But for hexagons that are close, vertex containment is a strong indicator
                # Let's add edge intersection check
                def segments_intersect(p1, p2, p3, p4):
                    def orient(a, b, c):
                        val = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                        if abs(val) < 1e-10:
                            return 0
                        return 1 if val > 0 else -1
                    
                    o1 = orient(p1, p2, p3)
                    o2 = orient(p1, p2, p4)
                    o3 = orient(p3, p4, p1)
                    o4 = orient(p3, p4, p2)
                    
                    if o1 != o2 and o3 != o4:
                        return True
                    return False
                
                n_i = len(verts_i)
                n_j = len(verts_j)
                for k in range(n_i):
                    for l in range(n_j):
                        if segments_intersect(verts_i[k], verts_i[(k+1)%n_i], verts_j[l], verts_j[(l+1)%n_j]):
                            return False
        return True

    def compute_min_outer_side(inner_hexes, outer_center, outer_angle):
        """Compute minimum outer side length for given inner hexagons and outer parameters."""
        # Binary search on outer side length
        lo, hi = 1.0, 50.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if all(is_hexagon_inside_outer(x, y, a, outer_center, mid, outer_angle) for x, y, a in inner_hexes):
                hi = mid
            else:
                lo = mid
        return hi

    # Candidate configurations to try
    best_side = 50.0
    best_config = None

    # Configuration 1: Linear arrangement (baseline)
    inner1 = [[2.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    side1 = compute_min_outer_side(inner1, (0.0, 0.0), 0.0)
    if side1 < best_side:
        best_side = side1
        best_config = (inner1, (0.0, 0.0), 0.0)

    if time.time() > deadline:
        return best_config[0], best_config[1], best_side, best_config[2]

    # Configuration 2: Honeycomb-like arrangement
    # Place hexagons in a hexagonal lattice pattern
    inner2 = []
    # Center
    inner2.append([0.0, 0.0, 0.0])
    # First ring: 6 hexagons
    for k in range(6):
        angle = math.radians(60 * k)
        x = 2.0 * math.cos(angle) * math.sqrt(3) / 2 * 2  # distance for non-overlap
        y = 2.0 * math.sin(angle) * math.sqrt(3) / 2 * 2
        # Actually, for two unit hexagons to touch, center distance = sqrt(3) if aligned
        # Let's use center distance = sqrt(3) for touching
        x = math.sqrt(3) * math.cos(angle)
        y = math.sqrt(3) * math.sin(angle)
        inner2.append([x, y, 0.0])
    # Second ring: 12 hexagons
    for k in range(12):
        angle = math.radians(30 + 30 * k)
        dist = 2 * math.sqrt(3)
        x = dist * math.cos(angle)
        y = dist * math.sin(angle)
        inner2.append([x, y, 0.0])
    # We have 1 + 6 + 12 = 19, but we need 11
    inner2 = inner2[:N]
    if check_non_overlap(inner2):
        side2 = compute_min_outer_side(inner2, (0.0, 0.0), 0.0)
        if side2 < best_side:
            best_side = side2
            best_config = (inner2, (0.0, 0.0), 0.0)

    if time.time() > deadline:
        return best_config[0], best_config[1], best_side, best_config[2]

    # Configuration 3: Rotated hexagonal arrangement
    inner3 = []
    inner3.append([0.0, 0.0, 0.0])
    for k in range(6):
        angle = math.radians(60 * k + 30)
        x = math.sqrt(3) * math.cos(angle)
        y = math.sqrt(3) * math.sin(angle)
        inner3.append([x, y, 0.0])
    for k in range(4):
        angle = math.radians(30 + 60 * k)
        dist = 2 * math.sqrt(3)
        x = dist * math.cos(angle)
        y = dist * math.sin(angle)
        inner3.append([x, y, 0.0])
    inner3 = inner3[:N]
    if check_non_overlap(inner3):
        side3 = compute_min_outer_side(inner3, (0.0, 0.0), 0.0)
        if side3 < best_side:
            best_side = side3
            best_config = (inner3, (0.0, 0.0), 0.0)

    if time.time() > deadline:
        return best_config[0], best_config[1], best_side, best_config[2]

    # Try optimizing the linear arrangement with slight rotations
    for rot_offset in [0, 15, 30, 45]:
        inner_rot = [[2.0 * k, 0.0, float(rot_offset)] for k in range(-(N // 2), N - N // 2)]
        if check_non_overlap(inner_rot):
            for outer_ang in [0, 15, 30]:
                side_r = compute_min_outer_side(inner_rot, (0.0, 0.0), outer_ang)
                if side_r < best_side:
                    best_side = side_r
                    best_config = (inner_rot, (0.0, 0.0), outer_ang)

    if time.time() > deadline:
        return best_config[0], best_config[1], best_side, best_config[2]

    # Try a dense rectangular grid arrangement
    # 3x4 grid with some adjustment
    inner4 = []
    cols = 4
    rows = 3
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= N:
                break
            x = (c - (cols - 1) / 2) * 2.0
            y = (r - (rows - 1) / 2) * math.sqrt(3)
            inner4.append([x, y, 0.0])
            idx += 1
    if len(inner4) == N and check_non_overlap(inner4):
        side4 = compute_min_outer_side(inner4, (0.0, 0.0), 0.0)
        if side4 < best_side:
            best_side = side4
            best_config = (inner4, (0.0, 0.0), 0.0)

    if time.time() > deadline:
        return best_config[0], best_config[1], best_side, best_config[2]

    # Try compacting the best config by optimizing positions
    if best_config is not None:
        inner_best = [list(h) for h in best_config[0]]
        oc_best = best_config[1]
        oa_best = best_config[2]
        
        # Simple local optimization: try small perturbations
        for _ in range(200):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                for dx in [-0.1, 0, 0.1]:
                    for dy in [-0.1, 0, 0.1]:
                        new_inner = [list(h) for h in inner_best]
                        new_inner[i][0] += dx
                        new_inner[i][1] += dy
                        if check_non_overlap(new_inner):
                            side_new = compute_min_outer_side(new_inner, oc_best, oa_best)
                            if side_new < best_side - 1e-6:
                                inner_best = new_inner
                                best_side = side_new
                                best_config = (inner_best, oc_best, oa_best)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break

    return best_config[0], best_config[1], best_side, best_config[2]
# EVOLVE-BLOCK-END
