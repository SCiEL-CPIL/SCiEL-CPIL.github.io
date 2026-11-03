"""Packing 12 unit hexagons in a regular hexagon, minimizing outer side length."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill

def hex_vertices(cx, cy, s, angle_deg):
    """Return vertices of a regular hexagon with center (cx,cy), side s, rotation angle_deg."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        verts.append((cx + s * np.cos(theta), cy + s * np.sin(theta)))
    return np.array(verts)

def polygon_contains_hexagon(outer_verts, inner_verts):
    """Check if outer polygon contains inner polygon using point-in-polygon for each inner vertex."""
    # For each inner vertex, check if it's inside the outer polygon
    for vx, vy in inner_verts:
        if not point_in_polygon(vx, vy, outer_verts):
            return False
    return True

def point_in_polygon(x, y, polygon):
    """Ray casting algorithm for point in polygon."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def polygons_disjoint_hex(hex1_verts, hex2_verts, eps=1e-9):
    """Check if two hexagons are disjoint (no interior overlap)."""
    # Check if any vertex of one is inside the other
    for v in hex1_verts:
        if point_in_polygon(v[0], v[1], hex2_verts):
            return False
    for v in hex2_verts:
        if point_in_polygon(v[0], v[1], hex1_verts):
            return False
    # Check edge intersections
    if edges_intersect_any(hex1_verts, hex2_verts):
        return False
    return True

def edges_intersect_any(verts1, verts2):
    """Check if any edge of polygon 1 intersects any edge of polygon 2."""
    for i in range(6):
        e1 = (verts1[i], verts1[(i+1)%6])
        for j in range(6):
            e2 = (verts2[j], verts2[(j+1)%6])
            if segments_intersect(e1, e2):
                return True
    return False

def segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def ccw(a, b, c):
        return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
    return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)

def compute_min_outer_side(inner_positions, inner_angles, outer_center, outer_angle):
    """
    Compute the minimum outer hexagon side length such that all inner hexagons fit.
    inner_positions: list of (x,y)
    inner_angles: list of angle_degrees
    outer_center: (cx, cy)
    outer_angle: rotation of outer hexagon
    Returns the minimum side length, or inf if invalid.
    """
    # For a regular hexagon with center (cx,cy), side s, rotation a:
    # The distance from center to each edge (apothem) = s * sqrt(3)/2
    # For a point (x,y) to be inside, its distance to each edge must be <= apothem
    
    outer_cx, outer_cy = outer_center
    a_outer = np.radians(outer_angle)
    
    # The 6 edge normals of the outer hexagon
    # Edge i is perpendicular to direction (a_outer + i*pi/3 + pi/2)
    # Actually, for a regular hexagon, the 6 edges have outward normals at angles
    # a_outer + pi/6 + i*pi/3 for i=0..5 (if vertices are at a_outer + i*pi/3)
    # Let me use: vertex i is at angle a_outer + i*pi/3
    # Edge between vertex i and vertex i+1 has outward normal at angle a_outer + (i+0.5)*pi/3
    
    # A point (x,y) is inside if for all 6 edges:
    # dot((x-cx, y-cy), normal_i) <= s * sqrt(3)/2
    # where normal_i is the unit outward normal of edge i
    
    apothem_factor = np.sqrt(3) / 2
    
    # Compute required s for each inner hexagon
    max_s = 0.0
    
    for (ix, iy), ia in zip(inner_positions, inner_angles):
        inner_verts = hex_vertices(ix, iy, 1.0, ia)
        for vx, vy in inner_verts:
            dx = vx - outer_cx
            dy = vy - outer_cy
            # For each of the 6 edge normals
            for i in range(6):
                # Normal direction of edge i
                normal_angle = a_outer + (i + 0.5) * np.pi / 3
                nx = np.cos(normal_angle)
                ny = np.sin(normal_angle)
                dist = dx * nx + dy * ny
                # Need dist <= s * apothem_factor, so s >= dist / apothem_factor
                if dist > 0:
                    s_needed = dist / apothem_factor
                    if s_needed > max_s:
                        max_s = s_needed
                else:
                    # dist <= 0 means the vertex is on the "inside" side of this edge's line
                    # which is fine, no constraint from this edge
                    pass
    
    return max_s

def evaluate_configuration(inner_positions, inner_angles, outer_center, outer_angle):
    """
    Evaluate a configuration. Returns (outer_side, is_valid).
    """
    # Check pairwise disjointness
    n = len(inner_positions)
    hex_verts_list = []
    for i in range(n):
        hex_verts_list.append(hex_vertices(inner_positions[i][0], inner_positions[i][1], 1.0, inner_angles[i]))
    
    for i in range(n):
        for j in range(i+1, n):
            if not polygons_disjoint_hex(hex_verts_list[i], hex_verts_list[j]):
                return (float('inf'), False)
    
    # Compute minimum outer side
    outer_side = compute_min_outer_side(inner_positions, inner_angles, outer_center, outer_angle)
    return (outer_side, True)

def try_arrangement(type_id, scale=1.0, offset=(0,0), inner_rot=0, outer_rot=0):
    """Try a specific arrangement type."""
    positions = []
    angles = []
    
    if type_id == 0:  # 2x6 grid
        for r in range(2):
            for c in range(6):
                x = (c - 2.5) * 2.0 * scale
                y = (r - 0.5) * np.sqrt(3) * scale
                positions.append((x + offset[0], y + offset[1]))
                angles.append(inner_rot)
    elif type_id == 1:  # 3x4 grid
        for r in range(3):
            for c in range(4):
                x = (c - 1.5) * 2.0 * scale
                y = (r - 1.0) * np.sqrt(3) * scale
                positions.append((x + offset[0], y + offset[1]))
                angles.append(inner_rot)
    elif type_id == 2:  # Hexagonal cluster: 1 center + 6 ring + 5 more
        # Center
        positions.append((offset[0], offset[1]))
        angles.append(inner_rot)
        # Ring of 6
        for i in range(6):
            theta = i * np.pi / 3
            x = offset[0] + 2.0 * scale * np.cos(theta)
            y = offset[1] + 2.0 * scale * np.sin(theta)
            positions.append((x, y))
            angles.append(inner_rot)
        # 5 more in second ring
        for i in range(5):
            theta = i * 2 * np.pi / 5 + np.pi/5
            x = offset[0] + 4.0 * scale * np.cos(theta)
            y = offset[1] + 4.0 * scale * np.sin(theta)
            positions.append((x, y))
            angles.append(inner_rot)
    elif type_id == 3:  # 4x3 grid (transposed)
        for r in range(4):
            for c in range(3):
                x = (c - 1.0) * 2.0 * scale
                y = (r - 1.5) * np.sqrt(3) * scale
                positions.append((x + offset[0], y + offset[1]))
                angles.append(inner_rot)
    elif type_id == 4:  # Two rows of 6, staggered
        for r in range(2):
            for c in range(6):
                x = (c - 2.5) * 2.0 * scale + (r * 1.0 * scale)
                y = (r - 0.5) * np.sqrt(3) * scale
                positions.append((x + offset[0], y + offset[1]))
                angles.append(inner_rot)
    elif type_id == 5:  # 6+6 in two hexagonal sub-clusters
        # Left cluster of 6
        for i in range(6):
            theta = i * np.pi / 3
            x = -2.0 * scale + offset[0] + 2.0 * scale * np.cos(theta)
            y = offset[1] + 2.0 * scale * np.sin(theta)
            positions.append((x, y))
            angles.append(inner_rot)
        # Right cluster of 6
        for i in range(6):
            theta = i * np.pi / 3
            x = 2.0 * scale + offset[0] + 2.0 * scale * np.cos(theta)
            y = offset[1] + 2.0 * scale * np.sin(theta)
            positions.append((x, y))
            angles.append(inner_rot)
    
    outer_center = (0.0, 0.0)
    outer_angle = outer_rot
    
    side, valid = evaluate_configuration(positions, angles, outer_center, outer_angle)
    return side, valid, positions, angles

def optimize_arrangement(type_id, init_positions, init_angles, init_outer_center, init_outer_angle):
    """Optimize an arrangement using scipy."""
    if time.time() - START_TIME > TIME_LIMIT:
        return None
    
    # Parameters: all positions (24), all angles (12), outer center (2), outer angle (1) = 39 params
    n = len(init_positions)
    x0 = []
    for pos in init_positions:
        x0.extend([pos[0], pos[1]])
    for ang in init_angles:
        x0.append(ang)
    x0.extend([init_outer_center[0], init_outer_center[1], init_outer_angle])
    
    def objective(params):
        if time.time() - START_TIME > TIME_LIMIT:
            return 1e10
        positions = []
        for i in range(n):
            positions.append((params[2*i], params[2*i+1]))
        angles = params[2*n : 2*n+n]
        outer_center = (params[2*n+n], params[2*n+n+1])
        outer_angle = params[2*n+n+2]
        
        side, valid = evaluate_configuration(positions, angles, outer_center, outer_angle)
        if not valid:
            return 1e10
        return side
    
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
    
    if time.time() - START_TIME > TIME_LIMIT:
        return None
    
    if result.fun < 1e9:
        positions = []
        for i in range(n):
            positions.append((result.x[2*i], result.x[2*i+1]))
        angles = result.x[2*n : 2*n+n]
        outer_center = (result.x[2*n+n], result.x[2*n+n+1])
        outer_angle = result.x[2*n+n+2]
        return (positions, angles, outer_center, outer_angle, result.fun)
    
    return None

def construct():
    best_side = float('inf')
    best_result = None
    
    # Try various arrangements
    arrangement_types = [0, 1, 2, 3, 4, 5]
    scales = [0.9, 1.0, 1.1]
    inner_rots = [0, 30]
    outer_rots = [0, 30]
    
    for atype in arrangement_types:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        for scale in scales:
            if time.time() - START_TIME > TIME_LIMIT:
                break
            for ir in inner_rots:
                if time.time() - START_TIME > TIME_LIMIT:
                    break
                for orr in outer_rots:
                    side, valid, positions, angles = try_arrangement(atype, scale, (0,0), ir, orr)
                    if valid and side < best_side:
                        best_side = side
                        best_result = (positions, angles, (0.0, 0.0), orr, side)
    
    # Now try to optimize the best few arrangements
    candidates_to_optimize = []
    for atype in arrangement_types[:3]:  # Only optimize top 3 types
        for scale in [1.0]:
            for ir in [0]:
                for orr in [0]:
                    side, valid, positions, angles = try_arrangement(atype, scale, (0,0), ir, orr)
                    if valid:
                        candidates_to_optimize.append((side, atype, positions, angles, orr))
    
    candidates_to_optimize.sort(key=lambda x: x[0])
    
    for side, atype, positions, angles, orr in candidates_to_optimize[:2]:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        opt_result = optimize_arrangement(atype, positions, angles, (0.0, 0.0), orr)
        if opt_result is not None:
            pos, ang, oc, oa, s = opt_result
            if s < best_side:
                best_side = s
                best_result = (pos, ang, oc, oa, s)
    
    # Final fallback
    if best_result is None:
        # Use the trivial arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    positions, angles, outer_center, outer_angle, side = best_result
    
    # Format output
    inner = [[pos[0], pos[1], ang] for pos, ang in zip(positions, angles)]
    
    return inner, outer_center, side, outer_angle

