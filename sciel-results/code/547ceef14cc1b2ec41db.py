import time
import math
import numpy as np
from itertools import combinations

N = 11

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, r, theta):
    """Vertices of regular hexagon centered at (cx,cy), circumradius r,
    with rotation theta (radians). Pointy-top at theta=0."""
    angles = theta + np.arange(6) * (np.pi / 3.0)
    return cx + r * np.cos(angles), cy + r * np.sin(angles)

def _hexagon_contains(outer_c, outer_s, outer_theta, inner_c, inner_theta):
    """Check if unit hexagon (circumradius 1) at inner_c with rotation inner_theta
    is fully inside outer hexagon (center outer_c, side outer_s, rotation outer_theta).
    Uses support functions: hexagon A contains B iff for all outer normals n_i,
    n_i . B <= n_i . A.
    
    For a regular hexagon with circumradius R, the inradius (apothem) is R*sqrt(3)/2.
    Side length s = R. Apothem a = s*sqrt(3)/2.
    Support function in direction n: a * max_i cos(angle_between_n_and_normal_i) ... 
    Actually for a convex polygon, h(n) = max_j n.v_j over vertices v_j.
    """
    # Outer hexagon vertices
    ox, oy = _hex_vertices(outer_c[0], outer_c[1], outer_s, outer_theta)
    # Inner hexagon vertices (circumradius 1)
    ix, iy = _hex_vertices(inner_c[0], inner_c[1], 1.0, inner_theta)
    
    # For containment, we need the inner hexagon inside the outer.
    # Equivalent: for each edge of outer hexagon, all inner vertices are on the correct side.
    # Edge i goes from vertex i to vertex (i+1)%6. The inward normal points toward center.
    for i in range(6):
        # Edge from (ox[i], oy[i]) to (ox[(i+1)%6], oy[(i+1)%6])
        ex = ox[(i+1)%6] - ox[i]
        ey = oy[(i+1)%6] - oy[i]
        # Inward normal (rotate edge direction by -90 degrees, normalize, ensure pointing inward)
        nx, ny = ey, -ex
        nlen = math.hypot(nx, ny)
        nx, ny = nx / nlen, ny / nlen
        # Check inward: normal should point toward center
        dx_c = outer_c[0] - ox[i]
        dy_c = outer_c[1] - oy[i]
        if nx * dx_c + ny * dy_c < 0:
            nx, ny = -nx, -ny
        # For all inner vertices, n . (v - p_i) >= -eps (on the inner side or on the line)
        # Actually n points inward, so n . (v - p_i) >= 0 for v inside
        for j in range(6):
            val = nx * (ix[j] - ox[i]) + ny * (iy[j] - oy[i])
            if val < -1e-9:
                return False
    return True

def _check_no_overlap(inner):
    """Check no two inner unit hexagons overlap. Two unit hexagons (same size, any rotation)
    are disjoint if distance between centers > 2 (circumradius sum) ... but that's sufficient not necessary.
    More precisely, for two convex polygons, check edge intersections and containment.
    Simple sufficient check: distance between centers >= 2 - epsilon means they might just touch.
    But rotated hexagons can have minimum distance less than circumradius sum.
    For safety, use distance > 1.999999 as a conservative check, or do proper polygon intersection.
    
    Actually, two unit regular hexagons have inradius sqrt(3)/2 ≈ 0.866.
    The minimum distance between two non-overlapping unit hexagons (any rotation) is at least 
    2 * inradius = sqrt(3) ≈ 1.732 when they touch edge-to-edge aligned.
    But with rotation, they can get closer. The minimum possible distance between centers of two
    non-overlapping unit hexagons is 2 * (inradius) = sqrt(3) when one vertex points at the other's flat side... 
    Actually the minimal separation is when they just touch: distance = 2 * apothem = sqrt(3) for aligned.
    With arbitrary rotation, the minimum could be less. Let's just use polygon intersection test.
    """
    n = len(inner)
    for i in range(n):
        for j in range(i + 1, n):
            cx1, cy1, t1 = inner[i]
            cx2, cy2, t2 = inner[j]
            # Quick reject by center distance
            dx, dy = cx2 - cx1, cy2 - cy1
            if math.hypot(dx, dy) > 2.0:
                continue  # definitely no overlap (max distance between points in hex is 2)
            # Get vertices
            x1, y1 = _hex_vertices(cx1, cy1, 1.0, t1)
            x2, y2 = _hex_vertices(cx2, cy2, 1.0, t2)
            # Check if any edge of hex1 intersects any edge of hex2
            # Or if any vertex of one is inside the other
            if _polygons_intersect(x1, y1, x2, y2):
                return False
    return True

def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def _point_in_polygon(px, py, verts_x, verts_y):
    """Ray casting point in polygon test."""
    n = len(verts_x)
    inside = False
    j = n - 1
    for i in range(n):
        if ((verts_y[i] > py) != (verts_y[j] > py)) and \
           (px < (verts_x[j] - verts_x[i]) * (py - verts_y[i]) / (verts_y[j] - verts_y[i]) + verts_x[i]):
            inside = not inside
        j = i
    return inside

def _polygons_intersect(x1, y1, x2, y2):
    """Check if two hexagons (given as vertex arrays) intersect or one contains the other."""
    # Check if any vertex of poly1 is inside poly2
    for i in range(6):
        if _point_in_polygon(x1[i], y1[i], x2, y2):
            return True
    # Check if any vertex of poly2 is inside poly1
    for i in range(6):
        if _point_in_polygon(x2[i], y2[i], x1, y1):
            return True
    # Check edge intersections
    for i in range(6):
        j = (i + 1) % 6
        for k in range(6):
            l = (k + 1) % 6
            if _segments_intersect((x1[i], y1[i]), (x1[j], y1[j]), (x2[k], y2[k]), (x2[l], y2[l])):
                return True
    return False

def _minimal_enclosing_hex_side(inner, outer_theta):
    """Find minimal side length of a regular hexagon with rotation outer_theta 
    that encloses all inner hexagons. We also optimize the center.
    
    For a given outer orientation theta, the constraint for inner hexagon i is:
    For each of the 6 outer normals n_k(theta), n_k . c_i + r_i(n_k) <= n_k . C + a
    where a = s * sqrt(3)/2 is the apothem, C is the outer center.
    
    So: s >= (2/sqrt(3)) * max_k [max_i (n_k . c_i + r_i(n_k)) - n_k . C]
    
    To minimize s over C, for each k we want n_k . C to be as large as possible (to reduce the max_i term's contribution).
    Actually: s >= (2/sqrt(3)) * max_k [ max_i (n_k . c_i + r_i(n_k)) - n_k . C ]
    
    Let M_k = max_i (n_k . c_i + r_i(n_k)). Then s >= (2/sqrt(3)) * max_k [M_k - n_k . C].
    
    To minimize max_k [M_k - n_k . C], this is a linear programming problem. 
    For simplicity, use the centroid of all inner centers as C, then refine.
    """
    # Outer normals: the 6 outward normals of the outer hexagon
    # For a hexagon with rotation theta, the outward normals are at angles theta + k*60 degrees
    # The apothem direction for face k is at angle theta + k*60
    normals = []
    for k in range(6):
        ang = outer_theta + k * (np.pi / 3.0)
        normals.append((math.cos(ang), math.sin(ang)))
    
    # For each inner hexagon, compute its support function in each normal direction
    # r_i(n) = max over vertices v of n . v
    supports = []  # supports[i][k] = n_k . (c_i + vertex_offset) = n_k . c_i + n_k . v_rel
    for (cx, cy, it) in inner:
        vx, vy = _hex_vertices(0, 0, 1.0, it)  # relative vertices
        row = []
        for (nx, ny) in normals:
            # n . (c + v) = nx*cx + ny*cy + nx*vx + ny*vy
            val = nx * cx + ny * cy + max(nx * vx[j] + ny * vy[j] for j in range(6))
            row.append(val)
        supports.append(row)
    
    # M_k = max over i of supports[i][k]
    M = [max(supports[i][k] for i in range(len(inner))) for k in range(6)]
    
    # Now minimize max_k [M_k - n_k . C] over C.
    # Use a simple iterative approach: start with centroid, then do a few gradient-like steps.
    cx0 = sum(p[0] for p in inner) / len(inner)
    cy0 = sum(p[1] for p in inner) / len(inner)
    C = (cx0, cy0)
    
    for _ in range(50):
        # Compute current max
        vals = [M[k] - (normals[k][0] * C[0] + normals[k][1] * C[1]) for k in range(6)]
        idx_max = np.argmax(vals)
        # Move C in direction of normal[idx_max] to increase n_k . C
        step = 0.1
        C_new = (C[0] + step * normals[idx_max][0], C[1] + step * normals[idx_max][1])
        vals_new = [M[k] - (normals[k][0] * C_new[0] + normals[k][1] * C_new[1]) for k in range(6)]
        if max(vals_new) < max(vals):
            C = C_new
        else:
            break
    
    max_val = max(M[k] - (normals[k][0] * C[0] + normals[k][1] * C[1]) for k in range(6))
    return (2.0 / math.sqrt(3.0)) * max_val, C

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin
    
    # Generate compact hexagonal cluster of N hexagons on triangular lattice
    # Triangular lattice: points (i*2 + j*1, j*sqrt(3)) or similar
    # Use axial coordinates for hex grid
    # Center at (0,0). Neighbors at 60-degree intervals, distance 2 (touching unit hexagons)
    
    def hex_distance(q, r):
        return (abs(q) + abs(r) + abs(q + r)) // 2
    
    # Generate lattice points in order of distance from center
    points = []
    max_radius = 5
    for q in range(-max_radius, max_radius + 1):
        for r in range(-max_radius, max_radius + 1):
            if hex_distance(q, r) <= max_radius:
                # Convert axial to cartesian: x = 2*q + r, y = r*sqrt(3)
                x = 2.0 * q + r
                y = r * math.sqrt(3)
                d = hex_distance(q, r)
                points.append((d, x, y, q, r))
    
    # Sort by distance, then by angle for determinism
    points.sort(key=lambda p: (p[0], p[1], p[2]))
    
    # Take the N closest points
    selected = points[:N]
    inner = [[x, y, 0.0] for (_, x, y, _, _) in selected]
    
    # Verify no overlap
    if not _check_no_overlap(inner):
        # Fallback: just spread them out
        inner = [[3.0 * k, 0.0, 0.0] for k in range(N)]
    
    # Find minimal enclosing hexagon by sweeping outer orientation
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_outer_theta = 0.0
    
    # Sweep theta from 0 to 60 degrees (hexagon has 60-degree symmetry)
    n_steps = 360
    for step in range(n_steps + 1):
        if time.time() > deadline - 10.0:
            break
        theta = (step / n_steps) * (math.pi / 3.0)
        s, c = _minimal_enclosing_hex_side(inner, theta)
        if s < best_side:
            best_side = s
            best_center = c
            best_outer_theta = theta
    
    # Refine with a finer sweep around the best theta
    if time.time() < deadline - 5.0:
        theta_center = best_outer_theta
        for _ in range(3):
            step_size = (math.pi / 3.0) / 3600
            best_local = best_side
            best_c = best_center
            best_t = best_outer_theta
            for k in range(-10, 11):
                t = theta_center + k * step_size
                s, c = _minimal_enclosing_hex_side(inner, t)
                if s < best_local:
                    best_local = s
                    best_c = c
                    best_t = t
            if best_local >= best_side:
                break
            best_side = best_local
            best_center = best_c
            best_outer_theta = best_t
            theta_center = best_outer_theta
    
    # Final verification
    outer_c = best_center
    outer_s = best_side
    outer_theta = best_outer_theta
    
    # Check validity
    for (cx, cy, it) in inner:
        if not _hexagon_contains(outer_c, outer_s, outer_theta, (cx, cy), it):
            # Expand slightly
            outer_s *= 1.001
            break
    
    if not _check_no_overlap(inner):
        # Emergency fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(N)]
        outer_s = 40.0
        outer_c = (0.0, 0.0)
        outer_theta = 0.0
    
    return inner, outer_c, outer_s, math.degrees(outer_theta)

# EVOLVE-BLOCK-END
