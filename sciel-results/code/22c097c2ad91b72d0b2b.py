"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    best_result = None
    best_side = float('inf')
    
    # Helper: compute bounding regular hexagon side length for a set of hexagons
    # Each inner hexagon: (x, y, angle_degrees)
    # Outer hexagon: center (cx, cy), side S, rotation angle A
    # We need to find minimum S such that all inner hexagons fit inside outer hexagon
    
    def hexagon_vertices(cx, cy, side, angle_deg):
        """Get vertices of regular hexagon."""
        verts = []
        for i in range(6):
            a = math.radians(angle_deg + i * 60)
            verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
        return verts
    
    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point is inside regular hexagon."""
        # Transform to hexagon's local frame
        dx = px - cx
        dy = py - cy
        a = math.radians(-angle_deg)
        lx = dx * math.cos(a) - dy * math.sin(a)
        ly = dx * math.sin(a) + dy * math.cos(a)
        
        # Regular hexagon with flat sides: |ly| <= side*sqrt(3)/2 and |lx| <= side
        # But it's actually: for each of 6 sides, check half-plane
        # Simpler: use the fact that a regular hexagon with side s has
        # width 2s and height s*sqrt(3) when flat-sided
        # In local coords: |lx| <= s and |ly| <= s*sqrt(3)/2 is NOT correct
        # The hexagon is: max over 6 directions of dot product <= s
        
        # The 6 outward normals of a regular hexagon (flat-top orientation)
        # For a hexagon centered at origin with side s, the support function is:
        # h(n) = s * max(|n_x|, |n_y * sqrt(3)/2 + |n_x|/2|, |n_y * sqrt(3)/2 - |n_x|/2|)
        # Simpler approach: check 6 half-planes
        
        # Vertices in local frame (flat-top, side s):
        # (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
        
        # Check if (lx, ly) is inside
        # The hexagon can be described as intersection of 6 half-planes
        # For flat-top hexagon with side s:
        # |ly| <= s * sqrt(3) / 2
        # |lx| + |ly| / sqrt(3) <= s  (for the slanted sides)
        
        s_val = side
        if abs(ly) > s_val * math.sqrt(3) / 2 + 1e-9:
            return False
        if abs(lx) + abs(ly) / math.sqrt(3) > s_val + 1e-9:
            return False
        return True
    
    def min_containing_hex_side(inner_hexes, outer_center, outer_angle):
        """Find minimum side length of outer hexagon to contain all inner hexagons."""
        # Binary search on side length
        lo = 0.0
        hi = 100.0
        
        def fits(S):
            for (ix, iy, iangle) in inner_hexes:
                # Check all 6 vertices of inner hexagon
                for j in range(6):
                    va = math.radians(iangle + j * 60)
                    vx = ix + math.cos(va)
                    vy = iy + math.sin(va)
                    if not point_in_hexagon(vx, vy, outer_center[0], outer_center[1], S, outer_angle):
                        return False
            return True
        
        if not fits(hi):
            return hi
        
        for _ in range(50):
            mid = (lo + hi) / 2
            if fits(mid):
                hi = mid
            else:
                lo = mid
        return hi
    
    # Try different arrangements
    # Arrangement 1: straight line
    inner_line = [(4.0 * k, 0.0, 0.0) for k in range(-(N // 2), N - N // 2)]
    side_line = min_containing_hex_side(inner_line, (0.0, 0.0), 0.0)
    if side_line < best_side:
        best_side = side_line
        best_result = (inner_line, (0.0, 0.0), side_line, 0.0)
    
    if time.time() > deadline:
        return best_result
    
    # Arrangement 2: hexagonal grid
    # Hexagonal tiling: each hexagon at position (col, row)
    # For flat-top hexagons: x = col * 1.5, y = row * sqrt(3) + (col % 2) * sqrt(3)/2
    # But we want unit hexagons (side=1)
    # Distance between centers in hex tiling: sqrt(3) for flat-top
    
    # Generate hexagonal cluster
    def generate_hex_cluster(n_hexes):
        """Generate positions for n_hexes in a hexagonal cluster."""
        positions = [(0, 0)]
        ring = 1
        while len(positions) < n_hexes:
            # Ring has 6*ring hexagons
            for side_idx in range(6):
                for step in range(ring):
                    # Direction vectors for hex grid
                    # Flat-top hex grid directions
                    if side_idx == 0:
                        x = ring + step
                        y = -step
                    elif side_idx == 1:
                        x = ring - step
                        y = ring
                    elif side_idx == 2:
                        x = -ring
                        y = ring - step
                    elif side_idx == 3:
                        x = -ring - step
                        y = step
                    elif side_idx == 4:
                        x = -ring + step
                        y = -ring
                    elif side_idx == 5:
                        x = ring
                        y = -ring + step
                    positions.append((x, y))
                    if len(positions) >= n_hexes:
                        break
                if len(positions) >= n_hexes:
                    break
            ring += 1
        return positions[:n_hexes]
    
    # Convert axial coordinates to Cartesian for flat-top hexagons
    # x_cart = sqrt(3) * (q + r/2)
    # y_cart = 3/2 * r
    def axial_to_cart(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y
    
    cluster_axial = generate_hex_cluster(N)
    inner_cluster = []
    for (q, r) in cluster_axial:
        x, y = axial_to_cart(q, r)
        inner_cluster.append((x, y, 0.0))
    
    side_cluster = min_containing_hex_side(inner_cluster, (0.0, 0.0), 0.0)
    if side_cluster < best_side:
        best_side = side_cluster
        best_result = (inner_cluster, (0.0, 0.0), side_cluster, 0.0)
    
    if time.time() > deadline:
        return best_result
    
    # Try rotated versions of the cluster
    for outer_angle in [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]:
        side_rot = min_containing_hex_side(inner_cluster, (0.0, 0.0), outer_angle)
        if side_rot < best_side:
            best_side = side_rot
            best_result = (inner_cluster, (0.0, 0.0), side_rot, outer_angle)
    
    if time.time() > deadline:
        return best_result
    
    # Try different inner rotations
    for inner_angle in [0.0, 30.0, 60.0]:
        inner_rot = [(x, y, inner_angle) for (x, y, _) in inner_cluster]
        for outer_angle in [0.0, 30.0, 60.0, 90.0]:
            side_r = min_containing_hex_side(inner_rot, (0.0, 0.0), outer_angle)
            if side_r < best_side:
                best_side = side_r
                best_result = (inner_rot, (0.0, 0.0), side_r, outer_angle)
            if time.time() > deadline:
                return best_result
    
    # Try compacting: shift centers to reduce bounding box
    # For the line arrangement, try different spacings
    for spacing in [math.sqrt(3), 1.5, 1.8, 2.0]:
        inner_spaced = [(spacing * k, 0.0, 0.0) for k in range(-(N // 2), N - N // 2)]
        side_s = min_containing_hex_side(inner_spaced, (0.0, 0.0), 0.0)
        if side_s < best_side:
            best_side = side_s
            best_result = (inner_spaced, (0.0, 0.0), side_s, 0.0)
        if time.time() > deadline:
            return best_result
    
    # Try 2D grid arrangements
    # For N=11, try arrangements like 3+4+4 or 2+3+3+3 etc.
    def try_grid_arrangement(rows):
        """rows is a list of counts per row."""
        if sum(rows) != N:
            return None
        inner = []
        y_offset = 0.0
        for i, count in enumerate(rows):
            x_offset = -(count - 1) * math.sqrt(3) / 2.0
            for j in range(count):
                x = x_offset + j * math.sqrt(3)
                y = y_offset
                inner.append((x, y, 0.0))
            y_offset += 1.5
        # Center the arrangement
        if inner:
            cx = sum(p[0] for p in inner) / len(inner)
            cy = sum(p[1] for p in inner) / len(inner)
            inner = [(p[0] - cx, p[1] - cy, p[2]) for p in inner]
        return inner
    
    # Try various row distributions
    row_configs = [
        [11],
        [5, 6],
        [6, 5],
        [4, 3, 4],
        [3, 5, 3],
        [2, 4, 3, 2],
        [2, 3, 4, 2],
        [3, 4, 4],
        [4, 4, 3],
        [1, 4, 4, 2],
        [2, 4, 4, 1],
        [3, 3, 3, 2],
        [2, 3, 3, 3],
        [1, 3, 4, 3],
        [3, 4, 3, 1],
        [2, 2, 4, 3],
        [3, 4, 2, 2],
        [1, 2, 4, 4],
        [4, 4, 2, 1],
        [2, 4, 2, 3],
        [3, 2, 4, 2],
        [1, 4, 3, 3],
        [3, 3, 4, 1],
        [2, 3, 3, 3],
        [3, 3, 3, 2],
        [1, 3, 3, 4],
        [4, 3, 3, 1],
        [2, 2, 3, 4],
        [4, 3, 2, 2],
        [1, 2, 3, 5],
        [5, 3, 2, 1],
        [2, 5, 3, 1],
        [1, 3, 5, 2],
        [2, 3, 5, 1],
        [1, 5, 3, 2],
        [3, 5, 1, 2],
        [2, 1, 5, 3],
        [3, 5, 2, 1],
        [1, 5, 2, 3],
        [2, 5, 1, 3],
        [3, 1, 5, 2],
        [2, 1, 3, 5],
        [5, 1, 3, 2],
        [2, 3, 1, 5],
        [5, 3, 1, 2],
        [1, 3, 2, 5],
        [5, 2, 3, 1],
        [2, 5, 3, 1],
        [1, 5, 3, 2],
    ]
    
    for config in row_configs:
        if time.time() > deadline:
            return best_result
        inner_arr = try_grid_arrangement(config)
        if inner_arr is None:
            continue
        for outer_angle in [0.0, 30.0]:
            side_a = min_containing_hex_side(inner_arr, (0.0, 0.0), outer_angle)
            if side_a < best_side:
                best_side = side_a
                best_result = (inner_arr, (0.0, 0.0), side_a, outer_angle)
    
    if best_result is None:
        # Fallback
        inner_line = [(4.0 * k, 0.0, 0.0) for k in range(-(N // 2), N - N // 2)]
        return inner_line, (0.0, 0.0), 40.0, 0.0
    
    return best_result
# EVOLVE-BLOCK-END
