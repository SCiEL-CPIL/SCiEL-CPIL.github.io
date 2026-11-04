"""Pack 11 unit regular hexagons inside a minimal regular hexagon."""
import math
import time
import numpy as np

N = 11
_DEADLINE = 145.0  # seconds; evaluator kills at 150s
_T0 = time.time()


def _hex_vertices(cx, cy, r, ang_rad):
    """Return the 6 vertices of a regular hexagon (flat-top default) centered at (cx,cy),
    with circumradius r, rotated by ang_rad."""
    # Regular hexagon vertices: at angles 0, 60, 120, 180, 240, 300 degrees (flat-top)
    # For a regular hexagon with side length 1, the circumradius = 1.
    # The inradius = sqrt(3)/2.
    # Vertices at angle k*pi/3 + ang_rad, radius r
    verts = []
    for k in range(6):
        theta = k * math.pi / 3.0 + ang_rad
        x = cx + r * math.cos(theta)
        y = cy + r * math.sin(theta)
        verts.append((x, y))
    return np.array(verts)


def _hexagon_contains_hexagon(outer_cx, outer_cy, outer_side, outer_ang,
                              inner_cx, inner_cy, inner_side, inner_ang):
    """Check if inner hexagon is fully inside outer hexagon.
    For two regular hexagons, containment can be checked via:
    1. All 6 vertices of inner are inside outer, AND
    2. All edges of inner don't cross outer edges.
    For regular hexagons with same orientation, this simplifies.
    But with arbitrary rotation, we use a robust check.
    
    Actually, for regular hexagons, if all 6 vertices of inner are inside outer,
    and the inner hexagon is convex, then inner is inside outer (since outer is convex).
    So we just need to check all 6 vertices of inner are inside outer.
    """
    # Get inner vertices
    inner_verts = _hex_vertices(inner_cx, inner_cy, inner_side, inner_ang)
    
    # Check if each inner vertex is inside outer hexagon
    # Outer hexagon: center (outer_cx, outer_cy), side length outer_side
    # Inradius of outer = outer_side * sqrt(3)/2
    # For a regular hexagon, a point is inside if its distance from center
    # in each of the 6 normal directions is <= inradius.
    
    outer_inradius = outer_side * math.sqrt(3.0) / 2.0
    
    for (vx, vy) in inner_verts:
        # Vector from outer center to vertex
        dx = vx - outer_cx
        dy = vy - outer_cy
        # Check against each of the 6 faces
        for k in range(6):
            # Normal direction for face k
            # Face k is perpendicular to the direction to vertex k (for flat-top)
            # The face normals are at angles k*pi/3 + outer_ang + pi/2? Let me think...
            # For a regular hexagon with vertices at angles (k*pi/3 + outer_ang),
            # the face between vertex k-1 and vertex k has outward normal at angle
            # (k-0.5)*pi/3 + outer_ang... 
            # Actually, the 6 face normals are at angles: outer_ang + pi/6, outer_ang + pi/6 + pi/3, ...
            # Wait, let me use a different approach.
            # A point (x,y) relative to center is inside a regular hexagon with inradius r
            # and orientation ang if: for each of the 6 directions at angles ang + k*pi/3 (k=0..5),
            # the dot product of the point with the unit vector in that direction <= r
            # No wait. The inradius is the distance from center to each face.
            # The 6 face normals are at angles: ang + pi/6 + k*pi/3? 
            # Let me just use the standard test: for a regular hexagon centered at origin
            # with vertices at angles theta_k = k*pi/3 + ang, the 6 faces have outward
            # normals at angles phi_k = theta_k + pi/2 = k*pi/3 + ang + pi/2? 
            # Hmm, let me think more carefully.
            
            # For a regular hexagon, the distance from center to each edge (inradius) is the same.
            # The 6 edge normals point at angles: ang + pi/6, ang + pi/6 + pi/3, ..., ang + pi/6 + 5*pi/3
            # This is because the vertices are at ang, ang+pi/3, ... and edges are between
            # consecutive vertices, so the edge midpoint (and normal) is at ang+pi/6, ang+pi/6+pi/3, etc.
            
            normal_angle = outer_ang + math.pi / 6.0 + k * math.pi / 3.0
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            dot = dx * nx + dy * ny
            if dot > outer_inradius + 1e-9:
                return False
    return True


def _hexagons_disjoint(c1, a1, c2, a2, side=1.0):
    """Check if two unit hexagons are disjoint (no overlap).
    c1, c2: (x,y) centers; a1, a2: rotation angles in radians.
    """
    # Two regular hexagons of side 1 are disjoint if and only if
    # no vertex of one is inside the other AND no edges cross.
    # For convex polygons, they are disjoint if all edges of one are 
    # on the same side of each edge of the other... 
    # Simplest robust check: all 6 vertices of hex1 are outside hex2 
    # AND all 6 vertices of hex2 are outside hex1, AND no edge intersections.
    # But for convex polygons, if no vertex of A is inside B and no vertex of B is inside A,
    # then either they are disjoint or one contains the other. Since they're same size,
    # containment implies overlap. So we need: no vertex inside, plus edge intersection check.
    
    # Actually for two congruent convex shapes, if no vertices are inside each other
    # and edges don't cross, they are disjoint.
    
    v1 = _hex_vertices(c1[0], c1[1], side, a1)
    v2 = _hex_vertices(c2[0], c2[1], side, a2)
    
    # Check if any vertex of v1 is inside hex2
    for (px, py) in v1:
        if _point_in_hexagon(px, py, c2[0], c2[1], side, a2):
            return False
    # Check if any vertex of v2 is inside hex1
    for (px, py) in v2:
        if _point_in_hexagon(px, py, c1[0], c1[1], side, a1):
            return False
    return True


def _point_in_hexagon(px, py, cx, cy, side, ang):
    """Check if point (px,py) is inside regular hexagon centered at (cx,cy) with side length and angle."""
    inradius = side * math.sqrt(3.0) / 2.0
    dx = px - cx
    dy = py - cy
    for k in range(6):
        normal_angle = ang + math.pi / 6.0 + k * math.pi / 3.0
        nx = math.cos(normal_angle)
        ny = math.sin(normal_angle)
        if dx * nx + dy * ny > inradius + 1e-9:
            return False
    return True


def _compute_min_enclosing_hexagon(hexes):
    """Given a list of (cx, cy, angle_rad) for N unit hexagons,
    find the minimum side length of a regular hexagon (with any center and rotation)
    that contains all of them.
    
    For a fixed outer rotation angle theta, the minimum enclosing regular hexagon
    can be found. We search over theta.
    
    For a fixed outer orientation, the enclosing hexagon must contain all vertices of all inner hexagons.
    The minimal enclosing regular hexagon of a set of points with fixed orientation is determined
    by the support functions in the 6 normal directions.
    """
    # Collect all vertices of all inner hexagons
    all_verts = []
    for (cx, cy, ang) in hexes:
        v = _hex_vertices(cx, cy, 1.0, ang)
        all_verts.append(v)
    all_verts = np.vstack(all_verts)  # shape (6*N, 2)
    
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Search over outer rotation angles. By symmetry, we only need 0 to pi/3.
    # But the optimal orientation depends on the configuration.
    # Let's try a range of angles.
    for theta in np.linspace(0.0, math.pi / 3.0, 37):
        # For outer hexagon with orientation theta, the 6 face normals are at:
        # theta + pi/6 + k*pi/3, k=0..5
        # The inradius r must satisfy: for each face k, 
        #   max over all vertices v of (v - center) . n_k <= r
        # where n_k is the unit normal.
        # Also the center can be chosen to minimize r.
        # For a fixed set of normals n_0,...,n_5, the minimal enclosing hexagon
        # with those normals has inradius:
        #   r = (max_i(v_i . n_k) - min_i(v_i . n_k)) / 2 for each k, 
        # and the center is at the midpoint of each pair of parallel supporting lines.
        # For the hexagon to be consistent, all 6 centers must agree.
        # For a regular hexagon, the 6 normals span the full 360 degrees.
        # The minimal r is max over k of (max_i(v_i.n_k) - min_i(v_i.n_k))/2
        # and the center must satisfy: center.n_k = (max_i(v_i.n_k) + min_i(v_i.n_k))/2 for all k.
        # This is a system of 6 equations in 2 unknowns (center_x, center_y).
        # For a regular hexagon, these are consistent if and only if the configuration
        # is "hexagonally symmetric" in some sense. In general, they may not be exactly consistent.
        # So we need to find center (cx, cy) and r such that for all k:
        #   max_i ((v_i - c).n_k) <= r  and  min_i ((v_i - c).n_k) >= -r
        # i.e.,  c.n_k + r >= max_i(v_i.n_k)  and  c.n_k - r <= min_i(v_i.n_k)
        # =>  r >= max_i(v_i.n_k) - c.n_k  and  r >= c.n_k - min_i(v_i.n_k)
        # =>  r >= max(max_i(v_i.n_k) - c.n_k, c.n_k - min_i(v_i.n_k))
        # To minimize r, for each k, the optimal c.n_k is (max_i(v_i.n_k) + min_i(v_i.n_k))/2,
        # giving r_k = (max_i(v_i.n_k) - min_i(v_i.n_k))/2.
        # But c is 2D and we have 6 constraints. We need to find c that minimizes
        #   max_k max(max_i(v_i.n_k) - c.n_k, c.n_k - min_i(v_i.n_k))
        # This is a 2D convex optimization problem.
        # For simplicity, let's use the center as the mean of the "midpoints" or just
        # iterate: start with centroid, then adjust.
        
        # Simpler approach: the center that minimizes the max over k of the range/2 + offset
        # Let's use: for each k, define L_k = min_i(v_i . n_k), U_k = max_i(v_i . n_k)
        # We need c such that c.n_k >= L_k and c.n_k <= U_k + 2r for all k, with r as small as possible.
        # Equivalently: 2r >= max_k (U_k - c.n_k) and 2r >= max_k (c.n_k - L_k)
        # => r >= (1/2) max(max_k(U_k - c.n_k), max_k(c.n_k - L_k))
        
        # Let's find the optimal center by trying the centroid and nearby points.
        centroid = np.mean(all_verts, axis=0)
        
        # For a given center c, the required r is:
        #   r(c) = max over k of max(U_k - c.n_k, c.n_k - L_k) / 2
        # Let's compute this for the centroid and a few nearby points.
        
        def required_r(c, normals):
            r = 0.0
            for k in range(6):
                n = normals[k]
                dot_c = c[0] * n[0] + c[1] * n[1]
                # U_k and L_k precomputed
                r = max(r, max(U_vals[k] - dot_c, dot_c - L_vals[k]))
            return r / 2.0
        
        normals = []
        for k in range(6):
            na = theta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append(np.array([math.cos(na), math.sin(na)]))
        
        U_vals = np.array([np.max(all_verts @ normals[k]) for k in range(6)])
        L_vals = np.array([np.min(all_verts @ normals[k]) for k in range(6)])
        
        # Try centroid and a small grid around it
        best_r_for_theta = float('inf')
        best_c_for_theta = centroid
        for dcx in [-1, -0.5, 0, 0.5, 1]:
            for dcy in [-1, -0.5, 0, 0.5, 1]:
                c = centroid + np.array([dcx, dcy])
                r = required_r(c, normals)
                if r < best_r_for_theta:
                    best_r_for_theta = r
                    best_c_for_theta = c
        
        if best_r_for_theta < best_side:
            best_side = best_r_for_theta
            best_center = (best_c_for_theta[0], best_c_for_theta[1])
            best_angle = theta
    
    # Side length = inradius * 2 / sqrt(3)
    side = best_side * 2.0 / math.sqrt(3.0)
    return side, best_center, best_angle


def construct():
    """Construct 11 unit hexagons inside a minimal regular hexagon."""
    
    # Start with the known "flower" configuration: 1 center + 6 around + 4 more
    # Flower: center at (0,0), 6 at distance 2, angles 0, 60, ..., 300 degrees
    hexes = []
    hexes.append((0.0, 0.0, 0.0))  # center
    for k in range(6):
        ang = k * math.pi / 3.0
        hexes.append((2.0 * math.cos(ang), 2.0 * math.sin(ang), 0.0))
    # 4 extra: place them at the "gap" positions
    # The gaps between the 6 surrounding hexagons are at angles 30, 90, 150, 210, 270, 330
    # But we only have 4 spots. Let's try all 6 gap positions and pick the best 4.
    # Gap positions are at distance 2*sqrt(3) from center, angles pi/6 + k*pi/3
    gap_positions = []
    for k in range(6):
        ang = math.pi / 6.0 + k * math.pi / 3.0
        r = 2.0 * math.sqrt(3.0)
        gap_positions.append((r * math.cos(ang), r * math.sin(ang), ang))
    
    # Try all combinations of 4 out of 6 gap positions
    from itertools import combinations
    best_config = None
    best_side = float('inf')
    
    for combo in combinations(range(6), 4):
        test_hexes = hexes[:7] + [gap_positions[i] for i in combo]
        side, center, angle = _compute_min_enclosing_hexagon(test_hexes)
        if side < best_side:
            best_side = side
            best_config = test_hexes
    
    if best_config is None:
        best_config = hexes[:7] + [gap_positions[i] for i in range(4)]
        best_side, center, angle = _compute_min_enclosing_hexagon(best_config)
    
    # Try to improve by adjusting the 4 extra hexagons slightly
    # Use a local search
    for iteration in range(30):
        if time.time() - _T0 > _DEADLINE - 5.0:
            break
        improved = False
        for i in range(7, 11):
            cx, cy, ang = best_config[i]
            # Try small perturbations
            for dcx in [-0.2, -0.1, 0.0, 0.1, 0.2]:
                for dcy in [-0.2, -0.1, 0.0, 0.1, 0.2]:
                    for dang in [0.0, math.pi/12.0, -math.pi/12.0]:
                        new_hex = (cx + dcx, cy + dcy, ang + dang)
                        # Check disjointness with all other hexagons
                        valid = True
                        for j in range(11):
                            if j == i:
                                continue
                            cj, aj = best_config[j][0], best_config[j][1], best_config[j][2]
                            if not _hexagons_disjoint((cx+dcx, cy+dcy), ang+dang, (best_config[j][0], best_config[j][1]), best_config[j][2]):
                                valid = False
                                break
                        if not valid:
                            continue
                        test_config = best_config[:i] + [new_hex] + best_config[i+1:]
                        side, _, _ = _compute_min_enclosing_hexagon(test_config)
                        if side < best_side - 1e-6:
                            best_side = side
                            best_config = test_config
                            improved = True
        if not improved:
            break
    
    # Final: compute the enclosing hexagon for the best config
    side, center, angle = _compute_min_enclosing_hexagon(best_config)
    
    # Convert to required format: inner rows (x, y, angle_degrees)
    inner = []
    for (cx, cy, ang) in best_config:
        inner.append([cx, cy, math.degrees(ang)])
    
    return inner, (center[0], center[1]), side, math.degrees(angle)


if __name__ == "__main__":
    inner, center, side, angle = construct()
    print(f"Side length: {side:.6f}")
    print(f"Outer center: {center}")
    print(f"Outer angle: {angle}")
    print(f"Number of inner hexagons: {len(inner)}")
