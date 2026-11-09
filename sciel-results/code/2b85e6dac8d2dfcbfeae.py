"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    sqrt3 = math.sqrt(3.0)
    
    def hex_vertices(cx, cy, angle_rad, side=1.0):
        """Return 6 vertices of a regular hexagon centered at (cx,cy) with given angle."""
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3.0
            verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
        return verts
    
    def point_in_hex(px, py, cx, cy, side, angle_rad):
        """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
        # Transform to local coordinates
        dx = px - cx
        dy = py - cy
        # Rotate by -angle
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        # For a regular hexagon with 'pointy-top' orientation (angle=0 gives flat-top),
        # the hexagon is defined by |x| <= s*sqrt(3)/2 and combinations
        # Actually, let's use the standard: for a hexagon with vertices at angles 0, 60, ..., 
        # the support function approach: max over 6 directions of dot product <= side * sqrt(3)/2
        # But easier: check against 6 edges
        # Vertices in local frame (angle=0 means first vertex at angle 0)
        # For our hex_vertices, angle_rad rotates the whole hex
        # In local frame (angle=0), vertices are at angles 0, 60, 120, 180, 240, 300
        # The hexagon is: max(|x|, |x*0.5 + y*sqrt(3)/2|, |-x*0.5 + y*sqrt(3)/2|) <= side * sqrt(3)/2
        # Wait, that's for flat-top. Let me just use the 6 edge test.
        
        # Vertices in local frame for angle=0:
        # (s, 0), (s/2, s*sqrt3/2), (-s/2, s*sqrt3/2), (-s, 0), (-s/2, -s*sqrt3/2), (s/2, -s*sqrt3/2)
        # This is a "pointy-right" hexagon.
        # Edge 1: from (s,0) to (s/2, s*sqrt3/2): direction (-s/2, s*sqrt3/2), normal (s*sqrt3/2, s/2) normalized -> (sqrt3/2, 1/2)
        # The constraint: dot((px,py), (sqrt3/2, 1/2)) <= s * sqrt3/2... 
        # Actually simpler: for a regular hexagon with inradius r = s*sqrt(3)/2,
        # the condition is that for all 6 directions at angles k*pi/3 + pi/6, 
        # dot(p, n_k) <= r
        # Let's use that.
        r = side * sqrt3 / 2.0
        for k in range(6):
            theta = k * math.pi / 3.0 + math.pi / 6.0
            nx = math.cos(theta)
            ny = math.sin(theta)
            if lx * nx + ly * ny > r + 1e-12:
                return False
        return True
    
    def hexes_overlap(v1, v2):
        """Check if two hexagons (lists of 6 points) overlap using edge intersection or containment."""
        # First check if any vertex of one is inside the other
        for p in v1:
            if point_in_hex(p[0], p[1], *center_of(v2), 1.0, angle_of(v2)):
                return True
        for p in v2:
            if point_in_hex(p[0], p[1], *center_of(v1), 1.0, angle_of(v1)):
                return True
        # Check edge-edge intersections
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i+1)%6]
            for j in range(6):
                p3 = v2[j]
                p4 = v2[(j+1)%6]
                if segments_intersect(p1, p2, p3, p4):
                    return True
        return False
    
    def center_of(verts):
        cx = sum(p[0] for p in verts) / 6.0
        cy = sum(p[1] for p in verts) / 6.0
        return cx, cy
    
    def angle_of(verts):
        # Recover angle from first vertex
        cx, cy = center_of(verts)
        if abs(verts[0][0]-cx) < 1e-15 and abs(verts[0][1]-cy) < 1e-15:
            return 0.0
        return math.atan2(verts[0][1]-cy, verts[0][0]-cx)
    
    def segments_intersect(p1, p2, p3, p4):
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
    
    def outer_contains_hex(outer_center, outer_side, outer_angle, hex_center, hex_angle):
        """Check if a unit hexagon is entirely inside the outer hexagon."""
        hex_verts = hex_vertices(hex_center[0], hex_center[1], hex_angle, 1.0)
        for vx, vy in hex_verts:
            if not point_in_hex(vx, vy, outer_center[0], outer_center[1], outer_side, outer_angle):
                return False
        return True
    
    def min_outer_side_for_config(positions, angles, outer_angle=0.0):
        """Find minimal outer side length (with given rotation) that contains all hexagons."""
        # The outer hexagon center should be chosen to minimize the required side.
        # For a given outer angle, the constraint is that all hexagon vertices must satisfy
        # the 6 half-plane inequalities of the outer hexagon.
        # The minimal side length is determined by the maximum of the support functions.
        
        # Collect all vertices
        all_verts = []
        for (cx, cy), ang in zip(positions, angles):
            verts = hex_vertices(cx, cy, ang, 1.0)
            all_verts.extend(verts)
        
        if not all_verts:
            return 0.0
        
        # The outer hexagon with center (ocx, ocy), side s, angle oa has vertices at
        # ocx + s*cos(oa + k*pi/3), ocy + s*sin(oa + k*pi/3) for k=0..5
        # A point (px, py) is inside iff for all k:
        #   (px-ocx)*cos(oa+k*pi/3+pi/6) + (py-ocy)*sin(oa+k*pi/3+pi/6) <= s*sqrt(3)/2
        
        # To minimize s given outer angle, we can optimize over center.
        # This is a convex optimization problem. For simplicity, try centering at centroid.
        cx0 = sum(p[0] for p in all_verts) / len(all_verts)
        cy0 = sum(p[1] for p in all_verts) / len(all_verts)
        
        # For this center, compute required s
        r_needed = 0.0
        for px, py in all_verts:
            for k in range(6):
                theta = outer_angle + k * math.pi / 3.0 + math.pi / 6.0
                val = (px - cx0) * math.cos(theta) + (py - cy0) * math.sin(theta)
                r_needed = max(r_needed, val)
        s = r_needed * 2.0 / sqrt3
        
        # Also try a few offset centers
        best_s = s
        for dx in [-0.5, 0, 0.5]:
            for dy in [-0.5, 0, 0.5]:
                cx1 = cx0 + dx
                cy1 = cy0 + dy
                r_needed = 0.0
                for px, py in all_verts:
                    for k in range(6):
                        theta = outer_angle + k * math.pi / 3.0 + math.pi / 6.0
                        val = (px - cx1) * math.cos(theta) + (py - cy1) * math.sin(theta)
                        r_needed = max(r_needed, val)
                s1 = r_needed * 2.0 / sqrt3
                if s1 < best_s:
                    best_s = s1
        
        return best_s
    
    # Initial configuration: hexagonal grid of 12 hexagons
    # Use a 4-3-3-2 arrangement or similar compact packing
    # For 12 hexagons, a good start is a hexagonal cluster of side 3 (which has 19) 
    # minus 7, or a rectangular-ish grid.
    
    # Let's use a grid: rows of 4, 4, 4 or a hexagonal pattern
    # Hexagonal grid positions for a compact cluster:
    # Row 0: 4 hexes
    # Row 1: 4 hexes offset
    # Row 2: 4 hexes
    
    positions = []
    angles = []
    
    # Create a 3x4 grid with hexagonal offset
    dx = 1.5  # horizontal spacing for flat-top hexes
    dy = sqrt3  # vertical spacing
    for row in range(3):
        for col in range(4):
            x = col * dx + (0.75 if row % 2 == 1 else 0.0)
            y = row * dy
            # Center the whole cluster
            positions.append((x, y))
            angles.append(0.0)
    
    # Center the cluster
    cx_mean = sum(p[0] for p in positions) / len(positions)
    cy_mean = sum(p[1] for p in positions) / len(positions)
    positions = [(p[0] - cx_mean, p[1] - cy_mean) for p in positions]
    
    # Greedy local search
    best_side = float('inf')
    best_positions = positions[:]
    best_angles = angles[:]
    best_outer_angle = 0.0
    
    # Try different outer angles
    for oa in [0, 30]:
        oa_rad = oa * math.pi / 180.0
        cur_positions = positions[:]
        cur_angles = angles[:]
        
        # Compute initial side
        cur_side = min_outer_side_for_config(cur_positions, cur_angles, oa_rad)
        
        # Local search: try moving each hexagon slightly
        step_sizes = [0.1, 0.5, 1.0, 2.0]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                for step in step_sizes:
                    for dx in [-step, 0, step]:
                        for dy in [-step, 0, step]:
                            if dx == 0 and dy == 0:
                                continue
                            new_pos = (cur_positions[i][0] + dx, cur_positions[i][1] + dy)
                            test_positions = cur_positions[:]
                            test_positions[i] = new_pos
                            
                            # Quick check: no overlaps with neighbors
                            overlap = False
                            for j in range(N):
                                if i == j:
                                    continue
                                if math.hypot(new_pos[0]-cur_positions[j][0], new_pos[1]-cur_positions[j][1]) < 2.0:
                                    v1 = hex_vertices(*new_pos, cur_angles[i], 1.0)
                                    v2 = hex_vertices(*cur_positions[j], cur_angles[j], 1.0)
                                    if hexes_overlap(v1, v2):
                                        overlap = True
                                        break
                            if overlap:
                                continue
                            
                            new_side = min_outer_side_for_config(test_positions, cur_angles, oa_rad)
                            if new_side < cur_side - 1e-6:
                                cur_positions = test_positions
                                cur_side = new_side
                                improved = True
                                if cur_side < best_side:
                                    best_side = cur_side
                                    best_positions = cur_positions[:]
                                    best_angles = cur_angles[:]
                                    best_outer_angle = oa_rad
        
        if cur_side < best_side:
            best_side = cur_side
            best_positions = cur_positions[:]
            best_angles = cur_angles[:]
            best_outer_angle = oa_rad
    
    # Format result
    inner = [[p[0], p[1], math.degrees(a)] for p, a in zip(best_positions, best_angles)]
    return inner, (0.0, 0.0), best_side, math.degrees(best_outer_angle)
# EVOLVE-BLOCK-END
