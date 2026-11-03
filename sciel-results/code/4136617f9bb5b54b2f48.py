"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    
    def get_hex_vertices(cx, cy, side, angle_deg):
        """Get vertices of a regular hexagon (flat-top when angle=0)."""
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices
    
    def hexagon_circumradius(side):
        """Circumradius of a regular hexagon with given side length."""
        return side  # For regular hexagon, circumradius = side
    
    def hexagon_apothem(side):
        """Apothem (inradius) of a regular hexagon with given side length."""
        return side * math.sqrt(3) / 2
    
    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
        # Transform to local coordinates
        dx = px - cx
        dy = py - cy
        angle = math.radians(angle_deg)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        lx = dx * cos_a + dy * sin_a
        ly = -dx * sin_a + dy * cos_a
        # For flat-top hexagon with circumradius r:
        # |lx| <= r and |ly| <= r*sqrt(3)/2 and |lx|/r + |ly|/(r*sqrt(3)/2) <= 1
        r = side
        if abs(lx) > r + 1e-9 or abs(ly) > r * math.sqrt(3)/2 + 1e-9:
            return False
        # Check the 6 half-planes
        for i in range(6):
            theta = i * math.pi / 3
            nx = math.cos(theta)
            ny = math.sin(theta)
            dist = lx * nx + ly * ny
            if dist > r + 1e-9:
                return False
        return True
    
    def hexagons_disjoint(hex1, hex2, tol=1e-6):
        """Check if two hexagons are disjoint (no overlap)."""
        x1, y1, a1 = hex1
        x2, y2, a2 = hex2
        # Get vertices
        v1 = get_hex_vertices(x1, y1, 1.0, a1)
        v2 = get_hex_vertices(x2, y2, 1.0, a2)
        
        # Check if any vertex of hex1 is inside hex2 or vice versa
        for vx, vy in v1:
            if point_in_hexagon(vx, vy, x2, y2, 1.0, a2):
                return False
        for vx, vy in v2:
            if point_in_hexagon(vx, vy, x1, y1, 1.0, a1):
                return False
        
        # Check edge intersections
        def segments_intersect(p1, p2, p3, p4):
            """Check if line segment p1-p2 intersects with p3-p4."""
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
        
        for i in range(6):
            for j in range(6):
                if segments_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                    return False
        return True
    
    def all_disjoint(inner_hexes, tol=1e-6):
        """Check if all hexagons are pairwise disjoint."""
        for i in range(len(inner_hexes)):
            for j in range(i+1, len(inner_hexes)):
                if not hexagons_disjoint(inner_hexes[i], inner_hexes[j], tol):
                    return False
        return True
    
    def all_inside_outer(inner_hexes, outer_center, outer_side, outer_angle):
        """Check if all inner hexagons are inside the outer hexagon."""
        ox, oy = outer_center
        for hx, hy, ha in inner_hexes:
            verts = get_hex_vertices(hx, hy, 1.0, ha)
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, ox, oy, outer_side, outer_angle):
                    return False
        return True
    
    def compute_minimal_outer_hex(inner_hexes, outer_center, outer_angle):
        """Compute the minimum outer hexagon side length to contain all inner hexagons."""
        ox, oy = outer_center
        # For a regular hexagon with center (ox,oy), side s, and angle,
        # a point (px,py) is inside if for all 6 directions, the projection
        # is within the apothem distance.
        # The apothem of the outer hexagon must be at least the maximum
        # projection of any inner hexagon vertex onto each of the 6 normals.
        
        angle = math.radians(outer_angle)
        max_apothem_needed = 0.0
        
        for hx, hy, ha in inner_hexes:
            verts = get_hex_vertices(hx, hy, 1.0, ha)
            for vx, vy in verts:
                dx = vx - ox
                dy = vy - oy
                # Transform to outer hexagon's local frame
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                lx = dx * cos_a + dy * sin_a
                ly = -dx * sin_a + dy * cos_a
                # For flat-top hexagon, the 6 normals are at angles 0, 60, 120, 180, 240, 300
                for i in range(6):
                    theta = i * math.pi / 3
                    nx = math.cos(theta)
                    ny = math.sin(theta)
                    proj = lx * nx + ly * ny
                    if proj > max_apothem_needed:
                        max_apothem_needed = proj
        
        # Apothem = side * sqrt(3) / 2, so side = apothem * 2 / sqrt(3)
        side = max_apothem_needed * 2.0 / math.sqrt(3)
        return side
    
    def evaluate_configuration(inner_hexes, outer_center, outer_angle):
        """Evaluate a configuration: return outer side length if valid, else infinity."""
        if not all_inside_outer(inner_hexes, outer_center, outer_angle):
            # Try to find the minimum outer hexagon that contains all
            pass
        side = compute_minimal_outer_hex(inner_hexes, outer_center, outer_angle)
        if not all_inside_outer(inner_hexes, outer_center, side, outer_angle):
            return float('inf')
        return side
    
    def honeycomb_arrangement(rows, cols, start_row=0):
        """Generate a honeycomb arrangement of hexagons."""
        hexes = []
        # For flat-top hexagons in honeycomb:
        # Same row: dx = 2*s, dy = 0
        # Adjacent row: dx = s, dy = sqrt(3)*s
        s = 1.0
        for r in range(rows):
            for c in range(cols):
                x = c * 2 * s + (r % 2) * s
                y = r * math.sqrt(3) * s
                hexes.append([x, y, 0.0])
        return hexes
    
    # Try several arrangements
    best_score = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Arrangement 1: 4x3 grid
    arrangements = []
    
    # 4x3 honeycomb
    hexes_4x3 = honeycomb_arrangement(3, 4)
    arrangements.append(hexes_4x3)
    
    # 3x4 honeycomb
    hexes_3x4 = honeycomb_arrangement(4, 3)
    arrangements.append(hexes_3x4)
    
    # 2x6 honeycomb
    hexes_2x6 = honeycomb_arrangement(2, 6)
    arrangements.append(hexes_2x6)
    
    # 6x2 honeycomb
    hexes_6x2 = honeycomb_arrangement(6, 2)
    arrangements.append(hexes_6x2)
    
    # 1x12 line
    hexes_1x12 = honeycomb_arrangement(1, 12)
    arrangements.append(hexes_1x12)
    
    # Hexagonal cluster: center + 6 around + 5 more
    # A perfect hexagonal number: 1+6=7, next would be 19. For 12, we can do:
    # Center, 6 neighbors, then 5 more in the next ring
    hexes_cluster = []
    s = 1.0
    # Center
    hexes_cluster.append([0.0, 0.0, 0.0])
    # First ring: 6 hexagons at distance 2*s in 6 directions
    for i in range(6):
        theta = i * math.pi / 3
        x = 2 * s * math.cos(theta)
        y = 2 * s * math.sin(theta)
        hexes_cluster.append([x, y, 0.0])
    # Second ring: add 5 more
    # The second ring positions for a hexagonal lattice
    second_ring_positions = []
    for i in range(6):
        theta = i * math.pi / 3
        # Positions at distance 4*s in 6 directions
        x = 4 * s * math.cos(theta)
        y = 4 * s * math.sin(theta)
        second_ring_positions.append((x, y))
        # Also positions at distance 2*sqrt(3)*s at 30, 90, 150, 210, 270, 330 degrees
        theta2 = i * math.pi / 3 + math.pi / 6
        x2 = 2 * math.sqrt(3) * s * math.cos(theta2)
        y2 = 2 * math.sqrt(3) * s * math.sin(theta2)
        second_ring_positions.append((x2, y2))
    # Add 5 of the 12 second ring positions
    for i in range(5):
        x, y = second_ring_positions[i]
        hexes_cluster.append([x, y, 0.0])
    arrangements.append(hexes_cluster)
    
    # More compact: try a 4-wide, 3-tall with offset
    hexes_compact = []
    for r in range(3):
        for c in range(4):
            x = c * 2.0 + (r % 2) * 1.0
            y = r * math.sqrt(3)
            hexes_compact.append([x, y, 0.0])
    arrangements.append(hexes_compact)
    
    # Try rotating the entire arrangement
    for arr in arrangements:
        # Center the arrangement
        xs = [h[0] for h in arr]
        ys = [h[1] for h in arr]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        
        # Try several outer angles
        for outer_angle in [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]:
            # Try several outer centers near the centroid
            for dx in [-2.0, -1.0, 0.0, 1.0, 2.0]:
                for dy in [-2.0, -1.0, 0.0, 1.0, 2.0]:
                    oc = (cx + dx, cy + dy)
                    
                    # Shift inner hexagons so centroid is at origin
                    shifted = [[h[0] - cx, h[1] - cy, h[2]] for h in arr]
                    
                    side = compute_minimal_outer_hex(shifted, (0.0, 0.0), outer_angle)
                    
                    # Verify validity
                    if all_inside_outer(shifted, (0.0, 0.0), side, outer_angle):
                        if side < best_score:
                            best_score = side
                            best_inner = shifted
                            best_outer_center = (0.0, 0.0)
                            best_outer_angle = outer_angle
    
    # If we found a valid configuration, return it
    if best_inner is not None:
        return best_inner, best_outer_center, best_score, best_outer_angle
    
    # Fallback: linear arrangement
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
# EVOLVE-BLOCK-END
