import math
import time

N = 11
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill

def is_time_up():
    return (time.time() - START_TIME) > TIME_LIMIT

def hexagon_vertices(cx, cy, side, angle_deg=0.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle_rad = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        vertices.append((x, y))
    return vertices

def min_enclosing_hexagon_side(points, cx, cy, angle_deg):
    """Find minimum side length of hexagon centered at (cx,cy) with given angle containing all points."""
    if not points:
        return 0.0
    
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(-angle_rad)
    sin_a = math.sin(-angle_rad)
    
    sqrt3 = math.sqrt(3)
    max_val = 0.0
    
    for px, py in points:
        dx = px - cx
        dy = py - cy
        # Rotate point by -angle to local coords
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        
        # Constraints for hexagon of side s centered at origin, vertex at (s,0):
        # 1. lx + ly/sqrt3 <= s
        # 2. ly * 2/sqrt3 <= s
        # 3. -lx + ly/sqrt3 <= s
        # 4. -lx - ly/sqrt3 <= s
        # 5. -ly * 2/sqrt3 <= s
        # 6. lx - ly/sqrt3 <= s
        
        v1 = lx + ly / sqrt3
        v2 = ly * 2 / sqrt3
        v3 = -lx + ly / sqrt3
        v4 = -lx - ly / sqrt3
        v5 = -ly * 2 / sqrt3
        v6 = lx - ly / sqrt3
        
        local_max = max(v1, v2, v3, v4, v5, v6)
        if local_max > max_val:
            max_val = local_max
            
    return max_val

def construct():
    best_side = float('inf')
    best_result = None
    
    # Generate the 11 most compact positions (hex grid)
    # Center (0,0)
    positions = [(0.0, 0.0)]
    
    # Ring 1: 6 neighbors
    # Flat-top hexagons. Neighbors at distance sqrt(3).
    # Angles 0, 60, 120, 180, 240, 300
    dist = math.sqrt(3)
    for i in range(6):
        theta = i * math.pi / 3
        x = dist * math.cos(theta)
        y = dist * math.sin(theta)
        positions.append((x, y))
        
    # Ring 2: Pick 4 closest
    # Candidates in Ring 2 (axial coordinates relative to center)
    # Axial (q,r) to pixel: x = 1.5*q, y = sqrt(3)*(r + q/2)
    # We need the 12 neighbors of the 6 ring 1 hexes that are not in ring 1 or center.
    # Easier: iterate axial coords for q,r in range -2 to 2, exclude q=0,r=0 and ring 1.
    # Ring 1 axial: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    
    ring2_candidates = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if q == 0 and r == 0:
                continue
            # Check if in ring 1
            if abs(q) <= 1 and abs(r) <= 1 and abs(q-r) <= 1:
                # This covers center and ring 1?
                # Center: (0,0)
                # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
                # Let's explicitly list ring 1 axial to be safe
                if (q,r) in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]:
                    continue
            # Calculate pixel
            x = 1.5 * q
            y = math.sqrt(3) * (r + q / 2.0)
            d2 = x*x + y*y
            ring2_candidates.append((d2, x, y))
            
    # Sort by distance
    ring2_candidates.sort(key=lambda item: item[0])
    
    # Take 4 closest
    for i in range(4):
        if i < len(ring2_candidates):
            d2, x, y = ring2_candidates[i]
            positions.append((x, y))
            
    # Ensure we have exactly 11
    if len(positions) > 11:
        positions = positions[:11]
    elif len(positions) < 11:
        # Fallback fill (should not happen)
        while len(positions) < 11:
            positions.append((0.0, 0.0))

    angles = [0.0] * 11
    
    # Get all vertices
    all_verts = []
    for (x, y), a in zip(positions, angles):
        verts = hexagon_vertices(x, y, 1.0, a)
        all_verts.extend(verts)
        
    # Centroid
    cx0 = sum(p[0] for p in positions) / 11.0
    cy0 = sum(p[1] for p in positions) / 11.0
    
    # Search for best outer hexagon
    # Try various angles
    # The cluster is symmetric, so angles 0-60 should cover it, but we'll check 0-360 in steps.
    # Step 1: Coarse search
    for outer_angle in range(0, 360, 5):
        if is_time_up():
            break
        side = min_enclosing_hexagon_side(all_verts, cx0, cy0, outer_angle)
        if side < best_side:
            best_side = side
            best_result = (positions, angles, (cx0, cy0), side, float(outer_angle))
            
    # Step 2: Fine search around best angle
    if best_result is not None:
        best_angle = best_result[4]
        # Refine in 0.5 deg steps
        for da in [-10, -9.5, -9, -8.5, -8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]:
            if is_time_up():
                break
            test_angle = (best_angle + da) % 360
            side = min_enclosing_hexagon_side(all_verts, cx0, cy0, test_angle)
            if side < best_side:
                best_side = side
                best_result = (positions, angles, (cx0, cy0), side, test_angle)
                
        # Try center offsets
        best_cx, best_cy = best_result[2]
        for dcx in [-0.2, -0.1, 0, 0.1, 0.2]:
            for dcy in [-0.2, -0.1, 0, 0.1, 0.2]:
                if is_time_up():
                    break
                cx_test = cx0 + dcx
                cy_test = cy0 + dcy
                side = min_enclosing_hexagon_side(all_verts, cx_test, cy_test, best_result[4])
                if side < best_side:
                    best_side = side
                    best_result = (positions, angles, (cx_test, cy_test), side, best_result[4])

    if best_result is None:
        # Fallback
        inner = [[0.0, 0.0, 0.0] for _ in range(11)]
        return inner, (0.0, 0.0), 10.0, 0.0

    positions, angles, center, side, outer_angle = best_result
    inner = [[x, y, a] for (x, y), a in zip(positions, angles)]
    return inner, center, float(side), float(outer_angle)
