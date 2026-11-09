import numpy as np
import time

N = 11

def hex_vertices(center, angle_deg, side=1.0):
    """Get vertices of a regular hexagon."""
    cx, cy = center
    angle = np.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = angle + k * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def point_in_hexagon(px, py, center, angle_deg, side):
    """Check if point (px,py) is inside hexagon."""
    cx, cy = center
    dx = px - cx
    dy = py - cy
    angle = np.radians(angle_deg)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    # Rotate point into hexagon's local frame
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a
    # Hexagon with vertex at angle 0 has constraints:
    # |lx| <= side
    # |ly| <= side * sqrt(3)/2
    # |lx|/2 + |ly|*sqrt(3)/2 <= side
    sqrt3 = np.sqrt(3)
    if abs(lx) > side + 1e-9:
        return False
    if abs(ly) > side * sqrt3 / 2 + 1e-9:
        return False
    if abs(lx) / 2 + abs(ly) * sqrt3 / 2 > side + 1e-9:
        return False
    return True

def min_dist_polygons(v1, v2):
    """Minimum distance between two convex polygons (arrays of vertices)."""
    min_d = np.inf
    n1, n2 = len(v1), len(v2)
    for i in range(n1):
        p1 = v1[i]
        p1n = v1[(i + 1) % n1]
        for j in range(n2):
            p2 = v2[j]
            p2n = v2[(j + 1) % n2]
            # Point to segment distance
            for p, a, b in [(p1, p1, p1n), (p2, p2, p2n)]:
                ab = b - a
                t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12)
                t = np.clip(t, 0, 1)
                closest = a + t * ab
                d = np.linalg.norm(p - closest)
                if d < min_d:
                    min_d = d
    return min_d

def hexagon_overlap_distance(c1, a1, c2, a2, side=1.0):
    """Return 0 if no overlap, negative if overlapping (penetration depth)."""
    v1 = hex_vertices(c1, a1, side)
    v2 = hex_vertices(c2, a2, side)
    # Check if any vertex of one is inside the other
    for v in v1:
        if point_in_hexagon(v[0], v[1], c2, a2, side):
            return -0.1  # overlapping
    for v in v2:
        if point_in_hexagon(v[0], v[1], c1, a1, side):
            return -0.1  # overlapping
    # Compute minimum distance
    d = min_dist_polygons(v1, v2)
    return d

def outer_hexagon_contains_point(px, py, ocx, ocy, oside, oangle_deg):
    """Check if point is inside outer hexagon."""
    dx = px - ocx
    dy = py - ocy
    angle = np.radians(oangle_deg)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a
    sqrt3 = np.sqrt(3)
    if abs(lx) > oside + 1e-9:
        return False
    if abs(ly) > oside * sqrt3 / 2 + 1e-9:
        return False
    if abs(lx) / 2 + abs(ly) * sqrt3 / 2 > oside + 1e-9:
        return False
    return True

def check_valid(inner, outer_center, outer_side, outer_angle):
    """Check if all inner hexagons are disjoint and inside outer hexagon."""
    ocx, ocy = outer_center
    # Check containment
    for x, y, ang in inner:
        verts = hex_vertices((x, y), ang, 1.0)
        for v in verts:
            if not outer_hexagon_contains_point(v[0], v[1], ocx, ocy, outer_side, outer_angle):
                return False
    # Check disjointness
    for i in range(len(inner)):
        for j in range(i + 1, len(inner)):
            xi, yi, ai = inner[i]
            xj, yj, aj = inner[j]
            d = hexagon_overlap_distance((xi, yi), ai, (xj, yj), aj)
            if d < -1e-6:  # overlapping
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Initial honeycomb-like arrangement
    # Center + 6 around + 4 more
    positions = []
    angles = []
    # Center
    positions.append([0.0, 0.0])
    angles.append(0.0)
    # 6 around center at distance sqrt(3)
    for k in range(6):
        theta = k * np.pi / 3
        positions.append([np.sqrt(3) * np.cos(theta), np.sqrt(3) * np.sin(theta)])
        angles.append(0.0)
    # 4 more: extend the honeycomb
    # Place at (2*sqrt(3), 0), (sqrt(3), 3), (sqrt(3), -3), (-2*sqrt(3), 0)
    positions.append([2 * np.sqrt(3), 0.0])
    angles.append(0.0)
    positions.append([np.sqrt(3), 3.0])
    angles.append(0.0)
    positions.append([np.sqrt(3), -3.0])
    angles.append(0.0)
    positions.append([-2 * np.sqrt(3), 0.0])
    angles.append(0.0)
    
    inner = [[positions[i][0], positions[i][1], angles[i]] for i in range(N)]
    
    # Force-directed optimization
    best_score = np.inf
    best_inner = None
    best_outer = None
    
    for iteration in range(2000):
        if time.time() > deadline - 10.0:
            break
        
        # Compute forces
        forces = np.zeros((N, 2))
        
        # Repulsive forces between overlapping hexagons
        for i in range(N):
            for j in range(i + 1, N):
                xi, yi, ai = inner[i]
                xj, yj, aj = inner[j]
                dx = xj - xi
                dy = yj - yi
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 2.0:  # Potential overlap
                    d = hexagon_overlap_distance((xi, yi), ai, (xj, yj), aj)
                    if d < 0.01:  # Overlapping or very close
                        # Push apart
                        if dist > 1e-6:
                            f = -5.0 * (0.01 - d) / dist
                            forces[i] += [dx * f, dy * f]
                            forces[j] += [-dx * f, -dy * f]
                        else:
                            # Random push
                            forces[i] += [0.1, 0.1]
                            forces[j] += [-0.1, -0.1]
        
        # Attractive force toward center
        for i in range(N):
            forces[i] += [-0.1 * inner[i][0], -0.1 * inner[i][1]]
        
        # Update positions
        for i in range(N):
            inner[i][0] += forces[i][0] * 0.05
            inner[i][1] += forces[i][1] * 0.05
        
        # Compute required outer side length
        ocx, ocy = 0.0, 0.0
        oangle = 0.0
        max_dist = 0.0
        for x, y, ang in inner:
            verts = hex_vertices((x, y), ang, 1.0)
            for v in verts:
                # Distance in hexagonal norm from outer center
                dx = v[0] - ocx
                dy = v[1] - ocy
                # For hexagon with vertex at angle 0, the "radius" in direction (dx,dy)
                # is the maximum t such that t*(dx,dy) is on the boundary
                # This is 1 / max(|cos|, |cos(60-theta)|, |cos(120-theta)|) type formula
                theta = np.arctan2(dy, dx)
                # The support function of hexagon with vertex at 0:
                # h(theta) = side * max(cos(0-theta), cos(60-theta), cos(120-theta), ...)
                # Actually for regular hexagon, h(theta) = side / max(1, 2*|sin(theta - k*60)|/sqrt(3)) for appropriate k
                # Simpler: the hexagon contains point t*(dx,dy) iff for all 6 edge normals n_k, t*(dx,dy)·n_k <= side
                # Edge normals are at angles 30, 90, 150, 210, 270, 330 degrees
                max_proj = 0
                for k in range(6):
                    nangle = np.radians(30 + k * 60)
                    nx = np.cos(nangle)
                    ny = np.sin(nangle)
                    proj = abs(dx * nx + dy * ny)
                    if proj > max_proj:
                        max_proj = proj
                if max_proj > 1e-9:
                    t = 1.0 / max_proj
                else:
                    t = 0
                if t > max_dist:
                    max_dist = t
        outer_side = max_dist
        
        if check_valid(inner, (ocx, ocy), outer_side, oangle):
            if outer_side < best_score:
                best_score = outer_side
                best_inner = [list(p) for p in inner]
                best_outer = (ocx, ocy, outer_side, oangle)
    
    if best_inner is not None:
        return best_inner, best_outer[0:2], best_outer[2], best_outer[3]
    
    # Fallback: return initial arrangement with computed outer size
    ocx, ocy = 0.0, 0.0
    oangle = 0.0
    max_dist = 0.0
    for x, y, ang in inner:
        verts = hex_vertices((x, y), ang, 1.0)
        for v in verts:
            dx = v[0] - ocx
            dy = v[1] - ocy
            max_proj = 0
            for k in range(6):
                nangle = np.radians(30 + k * 60)
                nx = np.cos(nangle)
                ny = np.sin(nangle)
                proj = abs(dx * nx + dy * ny)
                if proj > max_proj:
                    max_proj = proj
            if max_proj > 1e-9:
                t = 1.0 / max_proj
            else:
                t = 0
            if t > max_dist:
                max_dist = t
    outer_side = max_dist
    return inner, (ocx, ocy), outer_side, oangle

