"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    sqrt3 = math.sqrt(3)
    
    def hexagon_vertices(cx, cy, angle_deg, side=1.0):
        """Get 6 vertices of a regular hexagon."""
        angle_rad = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle_rad + math.radians(60 * i)
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices
    
    def point_in_hexagon(px, py, cx, cy, angle_deg, side=1.0):
        """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
        dx = px - cx
        dy = py - cy
        angle_rad = -math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        
        # For a regular hexagon with circumradius R, the inradius is R*sqrt(3)/2
        # In local coords, check against 6 half-planes
        max_dot = 0
        for k in range(6):
            theta = math.radians(30 + 60 * k)
            dot = lx * math.cos(theta) + ly * math.sin(theta)
            if dot > max_dot:
                max_dot = dot
        return max_dot <= side * sqrt3 / 2 + 1e-9
    
    def outer_hexagon_contains_inner(px, py, angle_deg, ocx, ocy, oangle_deg, oside):
        """Check if inner hexagon is inside outer hexagon."""
        verts = hexagon_vertices(px, py, angle_deg)
        for vx, vy in verts:
            if not point_in_hexagon(vx, vy, ocx, ocy, oangle_deg, oside):
                return False
        return True
    
    def segments_intersect(p1, p2, p3, p4):
        """Check if line segment p1-p2 intersects with line segment p3-p4."""
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
    
    def hexagons_disjoint(x1, y1, a1, x2, y2, a2):
        """Check if two unit hexagons are disjoint (no interior overlap)."""
        # Quick rejection: if center distance > 2, they're definitely disjoint
        dx = x1 - x2
        dy = y1 - y2
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > 2.0 + 1e-9:
            return True
        
        # If centers are very close, they definitely overlap
        if dist < 1e-9:
            return False
        
        # Get vertices
        v1 = hexagon_vertices(x1, y1, a1)
        v2 = hexagon_vertices(x2, y2, a2)
        
        # Check if any vertex of one is inside the other
        for vx, vy in v1:
            if point_in_hexagon(vx, vy, x2, y2, a2):
                return False
        for vx, vy in v2:
            if point_in_hexagon(vx, vy, x1, y1, a1):
                return False
        
        # Check edge intersections
        for i in range(6):
            for j in range(6):
                if segments_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                    return False
        
        return True
    
    def check_valid(inner, ocx, ocy, oside, oangle_deg):
        """Check if all inner hexagons are inside outer and don't overlap."""
        # Check containment
        for x, y, a in inner:
            if not outer_hexagon_contains_inner(x, y, a, ocx, ocy, oangle_deg, oside):
                return False
        
        # Check non-overlap
        n = len(inner)
        for i in range(n):
            for j in range(i + 1, n):
                if not hexagons_disjoint(inner[i][0], inner[i][1], inner[i][2],
                                         inner[j][0], inner[j][1], inner[j][2]):
                    return False
        return True
    
    def compute_min_outer_hexagon(positions, angles, center):
        """Compute minimum outer hexagon side length and orientation."""
        cx, cy = center
        best_side = float('inf')
        best_angle = 0.0
        
        # Try different orientations of the outer hexagon
        for oangle in range(0, 36, 2):  # 0 to 34 degrees, step 2
            # For each orientation, compute required side length
            # The outer hexagon must contain all vertices of all inner hexagons
            max_proj = 0
            for (px, py), a in zip(positions, angles):
                verts = hexagon_vertices(px, py, a)
                for vx, vy in verts:
                    # Project onto the 6 normal directions of the outer hexagon
                    dx = vx - cx
                    dy = vy - cy
                    for k in range(6):
                        theta = math.radians(oangle + 30 + 60 * k)
                        proj = abs(dx * math.cos(theta) + dy * math.sin(theta))
                        if proj > max_proj:
                            max_proj = proj
            
            # Required circumradius = max_proj / (sqrt(3)/2) = max_proj * 2/sqrt(3)
            required_side = max_proj * 2.0 / sqrt3
            
            if required_side < best_side:
                best_side = required_side
                best_angle = oangle
        
        return best_side, best_angle
    
    # Generate candidate positions in a honeycomb-like pattern
    # For unit hexagons with side 1, adjacent centers in honeycomb are sqrt(3) apart
    # in the direction perpendicular to shared edges
    
    # Let's try several compact arrangements
    # Arrangement 1: Linear chain
    # Arrangement 2: Hexagonal cluster (1 center + 6 around + 4 more)
    # Arrangement 3: Triangular grid
    # Arrangement 4: Random optimization
    
    best_result = None
    best_score = float('inf')
    
    # Try hexagonal cluster: center + 6 neighbors + 4 more
    # In honeycomb, 6 neighbors are at distance sqrt(3) from center
    # at angles 0, 60, 120, 180, 240, 300 degrees
    
    # Generate a set of candidate base positions
    base_positions_list = []
    
    # Pattern 1: Hexagonal cluster
    # Center at (0,0), 6 neighbors at distance sqrt(3)
    hex_cluster = [(0, 0)]
    for k in range(6):
        theta = math.radians(60 * k)
        hex_cluster.append((sqrt3 * math.cos(theta), sqrt3 * math.sin(theta)))
    
    # Add 4 more positions - try various configurations
    # Positions at distance 2*sqrt(3) from center in various directions
    for k in range(6):
        theta = math.radians(60 * k + 30)
        base_positions_list.append([(0, 0)] + [
            (sqrt3 * math.cos(math.radians(60 * j)), sqrt3 * math.sin(math.radians(60 * j)))
            for j in range(6)
        ] + [
            (2*sqrt3 * math.cos(theta), 2*sqrt3 * math.sin(theta)),
            (2*sqrt3 * math.cos(math.radians(60 * ((k+1)%6))), 2*sqrt3 * math.sin(math.radians(60 * ((k+1)%6)))),
            (2*sqrt3 * math.cos(math.radians(60 * ((k+5)%6))), 2*sqrt3 * math.sin(math.radians(60 * ((k+5)%6)))),
            (2*sqrt3 * math.cos(math.radians(60 * ((k+2)%6))), 2*sqrt3 * math.sin(math.radians(60 * ((k+2)%6)))),
        ])
    
    # Pattern 2: Two rows
    # Row 1: 6 hexagons, Row 2: 5 hexagons offset
    row_spacing = sqrt3  # vertical distance between rows
    col_spacing = 2.0  # horizontal distance within row
    
    for offset in [-1, 0, 1]:
        row1 = []
        for i in range(6):
            row1.append((i * col_spacing - 2.5 * col_spacing, 0))
        row2 = []
        for i in range(5):
            row2.append((i * col_spacing - 2.0 * col_spacing + 0.5 * col_spacing, row_spacing))
        base_positions_list.append(row1 + row2)
    
    # Pattern 3: Triangular arrangement
    # 4-3-4 pattern
    tri = []
    for i in range(4):
        tri.append((i * col_spacing - 1.5 * col_spacing, 0))
    for i in range(3):
        tri.append((i * col_spacing - 1.0 * col_spacing + 0.5 * col_spacing, row_spacing))
    for i in range(4):
        tri.append((i * col_spacing - 1.5 * col_spacing, 2 * row_spacing))
    base_positions_list.append(tri)
    
    # Pattern 4: 5-3-3 pattern
    p4 = []
    for i in range(5):
        p4.append((i * col_spacing - 2.0 * col_spacing, 0))
    for i in range(3):
        p4.append((i * col_spacing - 1.0 * col_spacing + 0.5 * col_spacing, row_spacing))
    for i in range(3):
        p4.append((i * col_spacing - 1.0 * col_spacing + 0.5 * col_spacing, 2 * row_spacing))
    base_positions_list.append(p4)
    
    # Pattern 5: Compact with some rotation
    # Try rotating some hexagons by 30 degrees for better packing
    for pattern_idx, positions in enumerate(base_positions_list):
        if len(positions) < N:
            continue
        
        # Take first N positions
        positions = positions[:N]
        
        # Try all with angle 0
        angles = [0.0] * N
        cx = sum(p[0] for p in positions) / N
        cy = sum(p[1] for p in positions) / N
        
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                if not hexagons_disjoint(positions[i][0], positions[i][1], angles[i],
                                        positions[j][0], positions[j][1], angles[j]):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            oside, oangle = compute_min_outer_hexagon(positions, angles, (cx, cy))
            if oside < best_score:
                best_score = oside
                inner = [[x, y, a] for (x, y), a in zip(positions, angles)]
                best_result = (inner, (cx, cy), oside, oangle)
        
        # Try with some hexagons rotated by 30 degrees
        if time.time() < deadline - 10:
            for rot_mask in range(min(8, 1 << N)):
                angles_rot = [0.0] * N
                for i in range(N):
                    if (rot_mask >> i) & 1:
                        angles_rot[i] = 30.0
                
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if not hexagons_disjoint(positions[i][0], positions[i][1], angles_rot[i],
                                                positions[j][0], positions[j][1], angles_rot[j]):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    oside, oangle = compute_min_outer_hexagon(positions, angles_rot, (cx, cy))
                    if oside < best_score:
                        best_score = oside
                        inner = [[x, y, a] for (x, y), a in zip(positions, angles_rot)]
                        best_result = (inner, (cx, cy), oside, oangle)
    
    # Try a simple optimization: start from best arrangement and perturb
    if best_result is not None and time.time() < deadline - 20:
        inner = best_result[0]
        ocx, ocy = best_result[1]
        
        # Local search: try small perturbations
        import random
        random.seed(42)
        
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            
            # Perturb a random hexagon
            trial = [list(x) for x in inner]
            idx = random.randint(0, N-1)
            trial[idx][0] += random.uniform(-0.1, 0.1)
            trial[idx][1] += random.uniform(-0.1, 0.1)
            
            # Check validity
            valid = True
            for i in range(N):
                if not outer_hexagon_contains_inner(trial[i][0], trial[i][1], trial[i][2],
                                                    ocx, ocy, best_result[3], best_score):
                    valid = False
                    break
                for j in range(i+1, N):
                    if not hexagons_disjoint(trial[i][0], trial[i][1], trial[i][2],
                                            trial[j][0], trial[j][1], trial[j][2]):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Recompute outer hexagon
                positions = [(x[0], x[1]) for x in trial]
                angles = [x[2] for x in trial]
                cx = sum(p[0] for p in positions) / N
                cy = sum(p[1] for p in positions) / N
                oside, oangle = compute_min_outer_hexagon(positions, angles, (cx, cy))
                
                if oside < best_score:
                    best_score = oside
                    best_result = (trial, (cx, cy), oside, oangle)
    
    if best_result is None:
        # Fallback: linear arrangement
        inner = [[2.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 20.0, 0.0
    
    return best_result
# EVOLVE-BLOCK-END

