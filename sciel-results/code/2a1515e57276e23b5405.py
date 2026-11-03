"""Packing 12 unit regular hexagons inside a regular hexagon, minimizing outer side length."""
import time
import math
import numpy as np

N = 12

# Helper: get vertices of a regular hexagon with given center, side length, rotation
def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle_rad + i * math.pi / 3.0
        x = cx + side * math.cos(a)
        y = cy + side * math.sin(a)
        verts.append((x, y))
    return verts

def _convex_hull(points):
    """Andrew's monotone chain convex hull. Returns list of (x,y) in CCW order."""
    points = sorted(set(points))
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
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

def _point_in_convex_polygon(point, poly):
    """Check if point is inside or on boundary of convex polygon (CCW order)."""
    n = len(poly)
    if n < 3:
        return True
    x, y = point
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        # Cross product to check if point is to the left of edge
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if cross < -1e-9:
            return False
    return True

def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def orient(a, b, c):
        val = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(val) < 1e-9:
            return 0
        return 1 if val > 0 else -1
    d1 = orient(p3, p4, p1)
    d2 = orient(p3, p4, p2)
    d3 = orient(p1, p2, p3)
    d4 = orient(p1, p2, p4)
    if d1 != d2 and d3 != d4:
        return True
    if d1 == 0 and _on_segment(p3, p4, p1):
        return True
    if d2 == 0 and _on_segment(p3, p4, p2):
        return True
    if d3 == 0 and _on_segment(p1, p2, p3):
        return True
    if d4 == 0 and _on_segment(p1, p2, p4):
        return True
    return False

def _on_segment(a, b, c):
    """Check if point c lies on segment a-b (assuming collinear)."""
    return (min(a[0], b[0]) - 1e-9 <= c[0] <= max(a[0], b[0]) + 1e-9 and
            min(a[1], b[1]) - 1e-9 <= c[1] <= max(a[1], b[1]) + 1e-9)

def _polygons_intersect(v1, v2):
    """Check if two convex polygons intersect using separating axis theorem."""
    # Check all edges of both polygons
    for poly in [v1, v2]:
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            # Normal vector
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            # Normal
            nx = -dy
            ny = dx
            # Project both polygons onto this normal
            min1, max1 = float('inf'), float('-inf')
            for p in v1:
                proj = p[0] * nx + p[1] * ny
                min1 = min(min1, proj)
                max1 = max(max1, proj)
            min2, max2 = float('inf'), float('-inf')
            for p in v2:
                proj = p[0] * nx + p[1] * ny
                min2 = min(min2, proj)
                max2 = max(max2, proj)
            # Check for separation
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return False
    # Check if one polygon is inside the other
    # Check if any vertex of v1 is inside v2 or vice versa
    if _point_in_convex_polygon(v1[0], v2):
        return True
    if _point_in_convex_polygon(v2[0], v1):
        return True
    # Check edge intersections
    for i in range(len(v1)):
        for j in range(len(v2)):
            if _segments_intersect(v1[i], v1[(i+1)%len(v1)], v2[j], v2[(j+1)%len(v2)]):
                return True
    return False

def _max_distance_to_center(verts):
    """Return max distance from origin to any vertex."""
    return max(math.sqrt(x*x + y*y) for x, y in verts)

def _check_feasibility(inner_positions, outer_side, outer_center, outer_angle):
    """Check if all inner hexagons fit inside outer hexagon and don't overlap."""
    ocx, ocy = outer_center
    outer_verts = _hex_vertices(ocx, ocy, outer_side, outer_angle)
    
    inner_verts_list = []
    for (x, y, ang) in inner_positions:
        verts = _hex_vertices(x, y, 1.0, ang)
        inner_verts_list.append(verts)
    
    # Check containment: all inner vertices must be inside outer hexagon
    for verts in inner_verts_list:
        for v in verts:
            if not _point_in_convex_polygon(v, outer_verts):
                return False
    
    # Check pairwise non-overlap
    for i in range(len(inner_verts_list)):
        for j in range(i+1, len(inner_verts_list)):
            if _polygons_intersect(inner_verts_list[i], inner_verts_list[j]):
                return False
    return True

def _min_outer_side_for_positions(inner_positions, outer_center, outer_angle):
    """Find minimum outer side length such that all inner hexagons fit inside."""
    ocx, ocy = outer_center
    # For a regular hexagon centered at (ocx, ocy) with rotation outer_angle,
    # the distance from center to each side (apothem) = side * sqrt(3)/2
    # The distance from center to each vertex = side
    # For a point to be inside, its distance from center in the direction perpendicular
    # to each side must be <= apothem.
    
    # The outer hexagon has 6 sides. For each side, the normal direction and distance
    # from center to side is the apothem.
    # A point (px, py) is inside if for all 6 sides, the projection onto the outward
    # normal is <= apothem.
    
    # Let's compute the required apothem for each inner vertex relative to outer center
    max_apothem_needed = 0.0
    for (x, y, ang) in inner_positions:
        verts = _hex_vertices(x, y, 1.0, ang)
        for (vx, vy) in verts:
            # Distance from outer center to this vertex
            dx = vx - ocx
            dy = vy - ocy
            # The 6 normal directions of the outer hexagon
            for k in range(6):
                theta = math.radians(outer_angle) + k * math.pi / 3.0
                # Outward normal direction
                nx = math.cos(theta)
                ny = math.sin(theta)
                proj = dx * nx + dy * ny
                # The apothem must be at least this projection (if positive)
                if proj > max_apothem_needed:
                    max_apothem_needed = proj
    
    # apothem = side * sqrt(3) / 2, so side = apothem * 2 / sqrt(3)
    if max_apothem_needed <= 0:
        return 0.0
    return max_apothem_needed * 2.0 / math.sqrt(3.0)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Strategy: Try various symmetric configurations and local optimization
    # Start with a grid-like arrangement and optimize
    
    best_side = 40.0
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Try different outer angles
    outer_angles = [0.0, 30.0]
    
    for outer_angle in outer_angles:
        # Initial guess: place hexagons in a hexagonal lattice pattern
        # For 12 hexagons, try a 3-4-3 or 2-4-2-4 arrangement
        
        # Configuration 1: Center + ring
        # Place 1 in center, 6 in first ring, 5 in second ring (partial)
        positions = []
        # Center
        positions.append((0.0, 0.0, 0.0))
        # First ring: 6 hexagons at distance 2 (side by side)
        for k in range(6):
            a = math.radians(60 * k)
            positions.append((2.0 * math.cos(a), 2.0 * math.sin(a), 0.0))
        # Second ring: 5 more
        for k in range(5):
            a = math.radians(60 * k + 30)
            positions.append((4.0 * math.cos(a), 4.0 * math.sin(a), 0.0))
        
        # Optimize using gradient descent on outer side length
        # We'll do a local search: perturb positions and check feasibility
        
        current_positions = [list(p) for p in positions]
        current_side = _min_outer_side_for_positions(current_positions, (0.0, 0.0), outer_angle)
        
        if current_side < best_side:
            best_side = current_side
            best_inner = [tuple(p) for p in current_positions]
            best_outer_center = (0.0, 0.0)
            best_outer_angle = outer_angle
        
        # Local optimization: try small perturbations
        step_size = 0.1
        for iteration in range(200):
            if time.time() > deadline - 10.0:
                break
            improved = False
            for i in range(len(current_positions)):
                for dx in [-step_size, 0.0, step_size]:
                    for dy in [-step_size, 0.0, step_size]:
                        for da in [-15.0, 0.0, 15.0]:
                            new_pos = [list(p) for p in current_positions]
                            new_pos[i][0] += dx
                            new_pos[i][1] += dy
                            new_pos[i][2] += da
                            
                            new_side = _min_outer_side_for_positions(new_pos, (0.0, 0.0), outer_angle)
                            
                            # Check if this configuration is valid (no overlaps)
                            # Quick check: ensure no two hexagons overlap
                            valid = True
                            new_verts = [_hex_vertices(p[0], p[1], 1.0, p[2]) for p in new_pos]
                            for a in range(len(new_verts)):
                                for b in range(a+1, len(new_verts)):
                                    if _polygons_intersect(new_verts[a], new_verts[b]):
                                        valid = False
                                        break
                                if not valid:
                                    break
                            
                            if valid and new_side < current_side - 1e-6:
                                current_positions = new_pos
                                current_side = new_side
                                improved = True
                                if current_side < best_side:
                                    best_side = current_side
                                    best_inner = [tuple(p) for p in current_positions]
                                    best_outer_center = (0.0, 0.0)
                                    best_outer_angle = outer_angle
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                step_size *= 0.7
                if step_size < 0.01:
                    break
        
        # Also try different initial configurations
        # Configuration 2: Two rows of 6
        positions2 = []
        for row in range(2):
            y = row * math.sqrt(3) - math.sqrt(3)/2
            for col in range(6):
                x = col * 2.0
                positions2.append((x, y, 0.0))
        
        current_positions2 = [list(p) for p in positions2]
        current_side2 = _min_outer_side_for_positions(current_positions2, (0.0, 0.0), outer_angle)
        
        if current_side2 < best_side:
            best_side = current_side2
            best_inner = [tuple(p) for p in current_positions2]
            best_outer_center = (0.0, 0.0)
            best_outer_angle = outer_angle
        
        # Local optimization for config 2
        step_size = 0.1
        for iteration in range(200):
            if time.time() > deadline - 10.0:
                break
            improved = False
            for i in range(len(current_positions2)):
                for dx in [-step_size, 0.0, step_size]:
                    for dy in [-step_size, 0.0, step_size]:
                        for da in [-15.0, 0.0, 15.0]:
                            new_pos = [list(p) for p in current_positions2]
                            new_pos[i][0] += dx
                            new_pos[i][1] += dy
                            new_pos[i][2] += da
                            
                            new_side = _min_outer_side_for_positions(new_pos, (0.0, 0.0), outer_angle)
                            
                            valid = True
                            new_verts = [_hex_vertices(p[0], p[1], 1.0, p[2]) for p in new_pos]
                            for a in range(len(new_verts)):
                                for b in range(a+1, len(new_verts)):
                                    if _polygons_intersect(new_verts[a], new_verts[b]):
                                        valid = False
                                        break
                                if not valid:
                                    break
                            
                            if valid and new_side < current_side2 - 1e-6:
                                current_positions2 = new_pos
                                current_side2 = new_side
                                improved = True
                                if current_side2 < best_side:
                                    best_side = current_side2
                                    best_inner = [tuple(p) for p in current_positions2]
                                    best_outer_center = (0.0, 0.0)
                                    best_outer_angle = outer_angle
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                step_size *= 0.7
                if step_size < 0.01:
                    break
    
    if best_inner is None:
        # Fallback: simple row
        best_inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0
        best_side = 40.0
    
    # Ensure we return the correct format
    inner_result = [[x, y, ang] for (x, y, ang) in best_inner]
    return inner_result, best_outer_center, best_side, best_outer_angle
