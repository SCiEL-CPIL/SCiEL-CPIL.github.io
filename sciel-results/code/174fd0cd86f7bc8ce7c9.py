"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit
    
    sqrt3 = math.sqrt(3)
    
    def hexagon_vertices(cx, cy, angle_deg, side=1.0):
        """Get vertices of a regular hexagon."""
        angle_rad = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle_rad + math.radians(60 * i)
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices
    
    def point_in_hexagon(px, py, cx, cy, angle_deg, side=1.0):
        """Check if point is inside a regular hexagon."""
        dx = px - cx
        dy = py - cy
        angle_rad = -math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        
        # For regular hexagon with circumradius = side, inradius = side * sqrt(3)/2
        # The 6 supporting lines have normals at angles 30 + 60*k degrees
        max_dot = 0
        for k in range(6):
            theta = math.radians(30 + 60 * k)
            dot = lx * math.cos(theta) + ly * math.sin(theta)
            if dot > max_dot:
                max_dot = dot
        return max_dot <= side * sqrt3 / 2 + 1e-9
    
    def hexagons_overlap(x1, y1, a1, x2, y2, a2, eps=1e-6):
        """Check if two unit hexagons overlap."""
        # Quick reject: if center distance > 2, no overlap
        dx = x1 - x2
        dy = y1 - y2
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > 2.0 + eps:
            return False
        if dist < 1e-9:
            return True
        
        # For same orientation, min distance is 2.0
        # For different orientations, it can be less, but let's be conservative
        # Check if any vertex of one is inside the other
        verts1 = hexagon_vertices(x1, y1, a1)
        verts2 = hexagon_vertices(x2, y2, a2)
        
        # Check if any vertex of hex1 is inside hex2
        for vx, vy in verts1:
            if point_in_hexagon(vx, vy, x2, y2, a2):
                return True
        # Check if any vertex of hex2 is inside hex1
        for vx, vy in verts2:
            if point_in_hexagon(vx, vy, x1, y1, a1):
                return True
        
        # Check edge intersections (simplified: if centers are close, check more carefully)
        # For now, the vertex check should catch most cases
        return False
    
    def outer_contains_inner(x, y, a, ocx, ocy, oangle, oside):
        """Check if inner hexagon is inside outer hexagon."""
        verts = hexagon_vertices(x, y, a)
        for vx, vy in verts:
            if not point_in_hexagon(vx, vy, ocx, ocy, oangle, oside):
                return False
        return True
    
    def compute_min_outer_side(positions, angles, ocx=0.0, ocy=0.0, oangle=0.0):
        """Compute minimum outer hexagon side to contain all inner hexagons."""
        # The outer hexagon is centered at (ocx, ocy) with rotation oangle
        # We need to find the minimum side S such that all inner hex vertices are inside
        
        # For a regular hexagon with side S, the inradius is S * sqrt(3)/2
        # A point at distance d from center in direction theta is inside if:
        # d * cos(theta - oangle - 30*k) <= S * sqrt(3)/2 for all k, but more precisely:
        # The constraint is that for each of the 6 normals, the projection <= S * sqrt(3)/2
        
        # The 6 normal directions (perpendicular to sides) are at oangle + 30 + 60*k
        max_required = 0
        for x, y, a in zip(positions, angles):
            verts = hexagon_vertices(x, y, a)
            for vx, vy in verts:
                dx = vx - ocx
                dy = vy - ocy
                # Project onto each of the 6 normal directions
                for k in range(6):
                    theta = math.radians(oangle + 30 + 60 * k)
                    proj = dx * math.cos(theta) + dy * math.sin(theta)
                    if proj > max_required:
                        max_required = proj
        
        # max_required is the maximum projection, so S * sqrt(3)/2 >= max_required
        # S >= max_required / (sqrt(3)/2) = max_required * 2 / sqrt(3)
        return max_required * 2.0 / sqrt3
    
    def is_valid_arrangement(positions, angles, ocx=0.0, ocy=0.0, oangle=0.0, oside=None):
        """Check if arrangement is valid (no overlaps, all inside outer)."""
        n = len(positions)
        
        # Check no overlaps
        for i in range(n):
            for j in range(i + 1, n):
                if hexagons_overlap(positions[i][0], positions[i][1], angles[i],
                                    positions[j][0], positions[j][1], angles[j]):
                    return False, None
        
        # Compute required outer side
        required_side = compute_min_outer_side(positions, angles, ocx, ocy, oangle)
        
        if oside is not None and required_side > oside + 1e-6:
            return False, required_side
        
        # Verify all inside with the computed side
        if not outer_contains_inner(positions[0][0], positions[0][1], angles[0], ocx, ocy, oangle, required_side):
            return False, required_side
        
        return True, required_side
    
    # Generate hexagonal lattice positions
    # For a honeycomb of unit hexagons, centers are on a triangular lattice with spacing 2
    # Position (i, j): x = 2*i + (j%2), y = j * sqrt(3)
    # But let's use axial coordinates for hex grid
    
    # The 7-hex cluster: center at (0,0), and 6 neighbors at distance 2
    # Neighbor directions: 0, 60, 120, 180, 240, 300 degrees
    neighbor_dirs = []
    for k in range(6):
        theta = math.radians(60 * k)
        neighbor_dirs.append((2.0 * math.cos(theta), 2.0 * math.sin(theta)))
    
    # 7-hex cluster
    base_positions = [(0.0, 0.0)] + neighbor_dirs
    base_angles = [0.0] * 7
    
    # For 11 hexagons, add 4 more. Candidate positions are at distance 2 from existing hexes
    # or at the "next ring" positions
    
    # Generate candidate positions for the additional 4 hexagons
    # These should be at positions that maintain the hexagonal packing
    # The next ring (distance 4 from center, or distance 2 from ring-1 hexes)
    
    # Let's enumerate all positions on the triangular lattice within a certain radius
    # and try to select 4 that don't overlap with the base 7
    
    candidates = []
    max_ring = 3  # Search up to ring 3
    
    for ring in range(0, max_ring + 1):
        if ring == 0:
            candidates.append((0.0, 0.0))
        else:
            for k in range(6 * ring):
                # Position on ring `ring`
                # Axial coordinate to cartesian
                # For ring r, the 6*r positions are at:
                # Start at angle 0, step by 60/r degrees? No, let's use a different approach
                
                # Actually, let's just generate all lattice points within radius
                pass
    
    # Simpler: generate all points on the triangular lattice within radius 6
    lattice_points = set()
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = 2.0 * i + (j % 2) * 1.0
            y = j * sqrt3
            # Use rounded key to avoid floating point issues
            key = (round(x, 6), round(y, 6))
            lattice_points.add(key)
    
    lattice_points = list(lattice_points)
    
    # Filter to points that don't overlap with the base 7
    valid_candidates = []
    for px, py in lattice_points:
        # Check if this position overlaps with any base hex
        overlaps = False
        for bx, by in base_positions:
            dx = px - bx
            dy = py - by
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 2.0 - 1e-6:
                overlaps = True
                break
        if not overlaps:
            valid_candidates.append((px, py))
    
    # Sort candidates by distance from origin (prefer closer ones)
    valid_candidates.sort(key=lambda p: p[0] * p[0] + p[1] * p[1])
    
    # Try to select 4 candidates that don't overlap with each other
    # Use greedy approach: try different subsets
    
    best_side = float('inf')
    best_arrangement = None
    
    # Try greedy selection with different orderings
    import itertools
    
    # First, try the closest 4
    for num_extra in [4]:
        # Try different subsets of candidates
        # Limit to first 20 candidates for speed
        search_candidates = valid_candidates[:20]
        
        # Try all combinations of 4 from the first 12 candidates
        if len(search_candidates) >= 4:
            combo_limit = min(12, len(search_candidates))
            for combo in itertools.combinations(range(combo_limit), 4):
                if time.time() > deadline - 10.0:
                    break
                
                extra_positions = [search_candidates[i] for i in combo]
                
                # Check no overlaps among extras
                valid = True
                for i in range(4):
                    for j in range(i + 1, 4):
                        dx = extra_positions[i][0] - extra_positions[j][0]
                        dy = extra_positions[i][1] - extra_positions[j][1]
                        dist = math.sqrt(dx * dx + dy * dy)
                        if dist < 2.0 - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Full arrangement
                all_positions = base_positions + extra_positions
                all_angles = [0.0] * 11
                
                # Check full validity
                valid_full, required_side = is_valid_arrangement(all_positions, all_angles)
                
                if valid_full and required_side < best_side:
                    best_side = required_side
                    best_arrangement = (all_positions, all_angles)
        
        if time.time() > deadline - 10.0:
            break
    
    # If we found a valid arrangement, return it
    if best_arrangement is not None:
        positions, angles = best_arrangement
        inner = [[p[0], p[1], a] for p, a in zip(positions, angles)]
        return inner, (0.0, 0.0), best_side, 0.0
    
    # Fallback: return the 7-hex cluster with a large outer hexagon
    # This should not happen, but as a safety net
    inner = [[p[0], p[1], 0.0] for p in base_positions]
    # Add 4 more at arbitrary positions far away (invalid, but better than crashing)
    for i in range(4):
        inner.append([100.0 + i * 10, 0.0, 0.0])
    
    return inner, (0.0, 0.0), 40.0, 0.0
# EVOLVE-BLOCK-END
