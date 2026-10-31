import numpy as np
import time

N = 11
SQRT3 = np.sqrt(3)
# Minimum center-to-center distance for disjoint unit hexagons (touching)
MIN_DIST = SQRT3

def get_hex_vertices(cx, cy, angle_deg):
    """Return 6 vertices of a regular unit hexagon."""
    angle_rad = np.deg2rad(angle_deg)
    angles = angle_rad + np.linspace(0, 2 * np.pi, 7)[:-1]
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def is_point_in_hexagon(pt, cx, cy, side, angle_deg):
    """Check if a point is inside a regular hexagon."""
    # Transform point to hexagon's local coordinate system
    dx = pt[0] - cx
    dy = pt[1] - cy
    angle_rad = -np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # Regular hexagon with side length s, vertices at angles 0, 60, 120, 180, 240, 300
    # For a regular hexagon with flat top/bottom (pointing right), the boundaries are:
    # |lx| <= s (along the x-axis)
    # |ly| <= s * sqrt(3) / 2 (along the y-axis)
    # But actually, a regular hexagon with vertices at (s,0), (s/2, s*sqrt(3)/2), etc.
    # The constraint is max(|lx|, |ly|*2/sqrt(3)) <= s
    # Wait, let me think again.
    
    # Vertices: (s*cos(0), s*sin(0)) = (s, 0)
    #            (s*cos(60), s*sin(60)) = (s/2, s*sqrt(3)/2)
    #            (s*cos(120), s*sin(120)) = (-s/2, s*sqrt(3)/2)
    #            etc.
    
    # The hexagon can be described by:
    # |lx| <= s
    # |ly| <= s * sqrt(3) / 2
    # This is not quite right either. Let me use the half-plane method.
    
    # The six edges of the hexagon (for vertices at 0, 60, 120, 180, 240, 300 degrees):
    # Edge 1: from (s,0) to (s/2, s*sqrt(3)/2). Normal pointing inward.
    # The line through (s,0) and (s/2, s*sqrt(3)/2) has equation:
    # x*cos(30) + y*sin(30) = s*cos(30)  ... let me derive properly.
    
    # For a regular hexagon with circumradius R = s (side length = circumradius for regular hexagon)
    # and vertices at angles 0, 60, 120, 180, 240, 300, the apothem (inradius) is s*sqrt(3)/2.
    # The six supporting lines have normals at angles 30, 90, 150, 210, 270, 330 degrees.
    # The constraint is: for each normal angle theta_k, lx*cos(theta_k) + ly*sin(theta_k) <= s*sqrt(3)/2
    
    normals = np.deg2rad([30, 90, 150, 210, 270, 330])
    projections = lx * np.cos(normals) + ly * np.sin(normals)
    return np.all(projections <= side * SQRT3 / 2)

def compute_required_side(inner, outer_center, outer_angle):
    """Compute the minimum outer side length to contain all inner hexagons."""
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    # For each inner hexagon, compute the required outer side to contain it
    # The outer hexagon has center (ox, oy), side length S, and rotation outer_angle
    
    # A point (px, py) is inside the outer hexagon iff, after transforming to local coords:
    # max over k of (lx*cos(theta_k) + ly*sin(theta_k)) <= S * sqrt(3) / 2
    
    normals = np.deg2rad([30, 90, 150, 210, 270, 330])
    cos_normals = np.cos(normals)
    sin_normals = np.sin(normals)
    
    max_projection = 0.0
    
    for (ix, iy, iangle) in inner:
        # Get all 6 vertices of this inner hexagon
        verts = get_hex_vertices(ix, iy, iangle)
        # Transform to outer hexagon's local coordinate system
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            angle_rad = -np.deg2rad(outer_angle)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            
            projections = lx * cos_normals + ly * sin_normals
            max_projection = max(max_projection, np.max(projections))
    
    # max_projection <= S * sqrt(3) / 2
    # S >= 2 * max_projection / sqrt(3)
    return 2.0 * max_projection / SQRT3

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Initial layout: hexagonal grid of unit hexagons
    # For N hexagons arranged in a hexagonal pattern
    # Let's use a simple linear arrangement first, then optimize
    
    # Place N unit hexagons in a line
    inner = [[i * SQRT3, 0.0, 0.0] for i in range(N)]
    outer_center = [0.0, 0.0]
    outer_angle = 0.0
    
    # Compute initial side length
    best_side = compute_required_side(inner, (0.0, 0.0), 0.0)
    
    # Try a few different initial arrangements
    arrangements = []
    
    # Linear arrangement
    arr1 = [[i * SQRT3, 0.0, 0.0] for i in range(N)]
    arrangements.append(arr1)
    
    # Two-row arrangement (staggered)
    arr2 = []
    for i in range(N):
        row = i // 2
        col = i % 2
        x = col * SQRT3 / 2 + row * SQRT3 * 3 / 2
        y = row * 3.0  # Wait, the vertical distance between rows of hexagons
        # For unit hexagons with flat top, the vertical spacing between rows is 3 (since height is 2*sqrt(3)/2 * 2 = sqrt(3)*... 
        # Actually, for a regular hexagon with side 1, the height (vertex to vertex) is 2, width (flat to flat) is sqrt(3)
        # So if we stack them, the y-spacing should be 3 for touching hexagons in a hex grid
        # Let me reconsider. Unit hexagon: circumradius = 1, inradius = sqrt(3)/2
        # Height (point to point) = 2, Width (flat to flat) = sqrt(3)
        # For a hexagonal tiling with pointy-top hexagons:
        # Horizontal spacing: sqrt(3), vertical spacing: 3/2 * 2 = 3? No.
        # Let me just use the fact that center-to-center distance must be >= sqrt(3) for disjoint unit hexagons.
        
        x = col * SQRT3 / 2
        y = row * 1.5 * 2  # This is getting complicated
    # Let me use a simpler approach: just use linear and a few variations
    
    # Try hexagonal packing: arrange in a hexagonal pattern
    # For N=11, we can do a hexagon with 1 in center, 6 in first ring, 4 more in second ring
    # Or just try different rotations and positions
    
    best_inner = arr1
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_side = compute_required_side(arr1, (0.0, 0.0), 0.0)
    
    # Try different outer hexagon rotations
    for outer_angle_deg in [0, 30, 60, 90, 120, 150]:
        side = compute_required_side(arr1, (0.0, 0.0), outer_angle_deg)
        if side < best_side:
            best_side = side
            best_outer_angle = outer_angle_deg
    
    # Try rotating the inner hexagons
    for inner_angle_deg in [0, 30, 60, 90, 120, 150]:
        arr_rotated = [[i * SQRT3, 0.0, inner_angle_deg] for i in range(N)]
        for outer_angle_deg in [0, 30, 60, 90, 120, 150]:
            side = compute_required_side(arr_rotated, (0.0, 0.0), outer_angle_deg)
            if side < best_side:
                best_side = side
                best_inner = arr_rotated
                best_outer_angle = outer_angle_deg
    
    # Now try a 2D arrangement
    # Place hexagons in a 2-row staggered pattern
    # For N hexagons, use ceil(N/2) in first row, floor(N/2) in second
    n1 = (N + 1) // 2
    n2 = N // 2
    
    for dx_offset in np.linspace(-SQRT3/4, SQRT3/4, 5):
        for dy_offset in [0.0, 0.5, 1.0]:
            arr_2d = []
            # Row 0
            total_width_1 = (n1 - 1) * SQRT3
            for i in range(n1):
                x = i * SQRT3 - total_width_1 / 2 + dx_offset
                y = dy_offset
                arr_2d.append([x, y, 0.0])
            # Row 1
            total_width_2 = (n2 - 1) * SQRT3
            for i in range(n2):
                x = i * SQRT3 - total_width_2 / 2 + SQRT3/4 + dx_offset
                y = -dy_offset  # Stagger vertically
                arr_2d.append([x, y, 0.0])
            
            # Check if any pair is too close
            valid = True
            for i in range(len(arr_2d)):
                for j in range(i + 1, len(arr_2d)):
                    dx = arr_2d[i][0] - arr_2d[j][0]
                    dy = arr_2d[i][1] - arr_2d[j][1]
                    if dx * dx + dy * dy < MIN_DIST * MIN_DIST - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                for outer_angle_deg in [0, 30, 60, 90, 120, 150]:
                    side = compute_required_side(arr_2d, (0.0, 0.0), outer_angle_deg)
                    if side < best_side:
                        best_side = side
                        best_inner = arr_2d
                        best_outer_angle = outer_angle_deg
    
    # Try 3-row arrangement
    n1 = (N + 2) // 3
    n2 = (N + 2) // 3
    n3 = N - n1 - n2
    
    if n3 >= 0:
        for dy1 in [0.0, 1.0, 1.5]:
            for dy2 in [0.0, 1.0, 1.5]:
                arr_3d = []
                # Row 0
                total_width_1 = max(0, n1 - 1) * SQRT3
                for i in range(n1):
                    x = i * SQRT3 - total_width_1 / 2
                    y = dy1
                    arr_3d.append([x, y, 0.0])
                # Row 1
                total_width_2 = max(0, n2 - 1) * SQRT3
                for i in range(n2):
                    x = i * SQRT3 - total_width_2 / 2 + SQRT3/4
                    y = 0.0
                    arr_3d.append([x, y, 0.0])
                # Row 2
                total_width_3 = max(0, n3 - 1) * SQRT3
                for i in range(n3):
                    x = i * SQRT3 - total_width_3 / 2
                    y = -dy2
                    arr_3d.append([x, y, 0.0])
                
                # Check distances
                valid = True
                for i in range(len(arr_3d)):
                    for j in range(i + 1, len(arr_3d)):
                        dx = arr_3d[i][0] - arr_3d[j][0]
                        dy = arr_3d[i][1] - arr_3d[j][1]
                        if dx * dx + dy * dy < MIN_DIST * MIN_DIST - 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    for outer_angle_deg in [0, 30, 60, 90, 120, 150]:
                        side = compute_required_side(arr_3d, (0.0, 0.0), outer_angle_deg)
                        if side < best_side:
                            best_side = side
                            best_inner = arr_3d
                            best_outer_angle = outer_angle_deg
    
    # Check time
    if time.time() > deadline:
        return best_inner, best_outer_center, best_side, best_outer_angle
    
    # Try to optimize the outer hexagon position
    # The outer hexagon should be centered on the bounding region of the inner hexagons
    # Compute the centroid of all inner hexagon vertices
    all_verts = []
    for (ix, iy, iangle) in best_inner:
        verts = get_hex_vertices(ix, iy, iangle)
        all_verts.extend(verts)
    all_verts = np.array(all_verts)
    centroid = np.mean(all_verts, axis=0)
    
    best_side2 = compute_required_side(best_inner, tuple(centroid), best_outer_angle)
    if best_side2 < best_side:
        best_side = best_side2
        best_outer_center = tuple(centroid)
    
    # Try a few more outer center positions
    for dx_c in np.linspace(-1, 1, 5):
        for dy_c in np.linspace(-1, 1, 5):
            oc = (centroid[0] + dx_c, centroid[1] + dy_c)
            side = compute_required_side(best_inner, oc, best_outer_angle)
            if side < best_side:
                best_side = side
                best_outer_center = oc
    
    return best_inner, best_outer_center, best_side, best_outer_angle
