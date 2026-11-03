import numpy as np
from scipy.optimize import minimize

N = 11

def hex_vertices(cx, cy, side, angle_deg):
    """Get vertices of a regular hexagon."""
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.arange(6) * np.pi / 3
    return cx + side * np.cos(angles), cy + side * np.sin(angles)

def hexagon_contains_point(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    # Transform point to hexagon's local frame
    dx = px - cx
    dy = py - cy
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # For a regular hexagon with side s, flat-to-flat distance is sqrt(3)*s
    # The hexagon contains points where max of |lx|, |ly|, |lx+ly|/sqrt(3) <= s
    # Actually, let's use the proper containment test
    # A regular hexagon with vertices at angles 0,60,...,300 has:
    # |x| <= s, |y| <= s*sqrt(3)/2, |x/2 + y*sqrt(3)/2| <= s
    # Wait, let me think more carefully.
    # For a hexagon with flat side on top (angle 0 pointing right):
    # The six edges are at distances s*sqrt(3)/2 from center in 6 directions
    # The containment is: for each of 6 directions, the projection <= s*sqrt(3)/2
    # Actually simpler: the hexagon is the intersection of 6 half-planes
    # Edge normals are at angles 30, 90, 150, 210, 270, 330 degrees
    # Each edge is at distance s*sqrt(3)/2 from center along the normal
    
    # Let me use a different approach: check if point is inside by checking
    # against all 6 edges
    vx, vy = hex_vertices(0, 0, side, angle_deg)
    vx = vx + cx
    vy = vy + cy
    
    # Point-in-polygon test
    inside = False
    n = 6
    for i in range(n):
        x1, y1 = vx[i], vy[i]
        x2, y2 = vx[(i+1) % n], vy[(i+1) % n]
        if ((y1 > py) != (y2 > py)) and (px < (x2 - x1) * (py - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside

def min_outer_side(inner_positions, outer_center, outer_angle_deg):
    """Find minimum outer hexagon side length that contains all inner hexagons."""
    # For each inner hexagon, find the maximum distance from outer center to any vertex
    # in the direction that constrains the outer hexagon
    # Actually, we need the outer hexagon (with given center and angle) to contain all inner hexagons
    # The outer hexagon with center C, side S, angle A contains a point P if and only if
    # for each of the 6 edge half-planes, P is on the correct side
    
    # The 6 edge normals of the outer hexagon are at angles A+30, A+90, A+150, A+210, A+270, A+330
    # Each edge is at distance S*sqrt(3)/2 from center
    
    # For the outer hexagon to contain an inner hexagon, all vertices of the inner hexagon
    # must be inside the outer hexagon.
    
    # For a given outer side length S, the constraint is:
    # For each inner vertex (vx, vy) and each of 6 normals n_i:
    #   dot((vx-cx, vy-cy), n_i) <= S*sqrt(3)/2
    
    # So S >= max over all inner vertices and all normals of dot((v-c), n_i) / (sqrt(3)/2)
    
    cx, cy = outer_center
    angle_rad = np.radians(outer_angle_deg)
    
    # The 6 outward normals of the outer hexagon
    normal_angles = angle_rad + np.pi/6 + np.arange(6) * np.pi / 3
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    max_proj = 0.0
    for (ix, iy, iangle) in inner_positions:
        vx, vy = hex_vertices(ix, iy, 1.0, iangle)
        for i in range(6):
            dx = vx - cx
            dy = vy - cy
            proj = dx * normals[i, 0] + dy * normals[i, 1]
            if proj > max_proj:
                max_proj = proj
    
    # S*sqrt(3)/2 >= max_proj, so S >= 2*max_proj/sqrt(3)
    return 2.0 * max_proj / np.sqrt(3)

def construct():
    # Generate various candidate configurations
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Try different arrangements of 11 hexagons
    # Arrangement types: (row_lengths, row_offset_pattern)
    
    arrangements = []
    
    # Single row
    arrangements.append({'rows': [11], 'type': 'linear'})
    
    # Two rows
    arrangements.append({'rows': [6, 5], 'type': 'two_row'})
    arrangements.append({'rows': [5, 6], 'type': 'two_row'})
    
    # Three rows
    arrangements.append({'rows': [4, 3, 4], 'type': 'three_row'})
    arrangements.append({'rows': [3, 5, 3], 'type': 'three_row'})
    arrangements.append({'rows': [4, 4, 3], 'type': 'three_row'})
    arrangements.append({'rows': [3, 4, 4], 'type': 'three_row'})
    arrangements.append({'rows': [5, 3, 3], 'type': 'three_row'})
    arrangements.append({'rows': [3, 3, 5], 'type': 'three_row'})
    arrangements.append({'rows': [2, 7, 2], 'type': 'three_row'})
    arrangements.append({'rows': [2, 5, 4], 'type': 'three_row'})
    arrangements.append({'rows': [4, 5, 2], 'type': 'three_row'})
    
    # Four rows
    arrangements.append({'rows': [3, 4, 3, 1], 'type': 'four_row'})
    arrangements.append({'rows': [2, 3, 4, 2], 'type': 'four_row'})
    arrangements.append({'rows': [3, 3, 3, 2], 'type': 'four_row'})
    arrangements.append({'rows': [2, 4, 3, 2], 'type': 'four_row'})
    arrangements.append({'rows': [1, 4, 4, 2], 'type': 'four_row'})
    arrangements.append({'rows': [2, 4, 4, 1], 'type': 'four_row'})
    arrangements.append({'rows': [1, 3, 5, 2], 'type': 'four_row'})
    arrangements.append({'rows': [2, 5, 3, 1], 'type': 'four_row'})
    arrangements.append({'rows': [1, 5, 3, 2], 'type': 'four_row'})
    
    # Five rows
    arrangements.append({'rows': [2, 3, 3, 2, 1], 'type': 'five_row'})
    arrangements.append({'rows': [1, 3, 3, 3, 1], 'type': 'five_row'})
    arrangements.append({'rows': [1, 2, 5, 2, 1], 'type': 'five_row'})
    arrangements.append({'rows': [1, 3, 4, 2, 1], 'type': 'five_row'})
    arrangements.append({'rows': [2, 2, 3, 3, 1], 'type': 'five_row'})
    
    # Hexagonal cluster: center + 6 around + 4 more
    arrangements.append({'rows': [1, 3, 5, 2], 'type': 'hex_cluster'})
    arrangements.append({'rows': [2, 5, 3, 1], 'type': 'hex_cluster'})
    
    for arr in arrangements:
        rows = arr['rows']
        if sum(rows) != N:
            continue
        
        # Generate positions for this arrangement
        # For honeycomb packing:
        # Row spacing: 1.5 (vertical distance between row centers)
        # Within a row: spacing sqrt(3) between centers
        # Odd/even rows offset by sqrt(3)/2
        
        sqrt3 = np.sqrt(3)
        positions = []
        
        num_rows = len(rows)
        # Center the arrangement vertically
        total_height = (num_rows - 1) * 1.5
        y_start = -total_height / 2.0
        
        for row_idx, count in enumerate(rows):
            y = y_start + row_idx * 1.5
            # Center this row horizontally
            row_width = (count - 1) * sqrt3
            x_start = -row_width / 2.0
            
            # Offset for alternating rows
            if row_idx % 2 == 1:
                x_offset = sqrt3 / 2.0
            else:
                x_offset = 0.0
            
            for col in range(count):
                x = x_start + col * sqrt3 + x_offset
                positions.append((x, y, 0.0))
        
        # Try different outer hexagon angles and centers
        for outer_angle in [0, 30, 60, 90, 120, 150]:
            # Try centering the outer hexagon at the centroid of inner hexagons
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            centroid_x = np.mean(xs)
            centroid_y = np.mean(ys)
            
            # Try several center offsets
            for dx in [-0.5, 0.0, 0.5]:
                for dy in [-0.5, 0.0, 0.5]:
                    outer_center = (centroid_x + dx, centroid_y + dy)
                    side = min_outer_side(positions, outer_center, outer_angle)
                    if side < best_side:
                        best_side = side
                        best_inner = [list(p) for p in positions]
                        best_outer_center = outer_center
                        best_outer_angle = outer_angle
    
    # Also try rotating inner hexagons
    # For each arrangement, try rotating all inner hexagons by 30 degrees
    for arr in arrangements:
        rows = arr['rows']
        if sum(rows) != N:
            continue
        
        sqrt3 = np.sqrt(3)
        positions = []
        
        num_rows = len(rows)
        total_height = (num_rows - 1) * 1.5
        y_start = -total_height / 2.0
        
        for row_idx, count in enumerate(rows):
            y = y_start + row_idx * 1.5
            row_width = (count - 1) * sqrt3
            x_start = -row_width / 2.0
            
            if row_idx % 2 == 1:
                x_offset = sqrt3 / 2.0
            else:
                x_offset = 0.0
            
            for col in range(count):
                x = x_start + col * sqrt3 + x_offset
                positions.append((x, y, 30.0))  # Rotated 30 degrees
        
        for outer_angle in [0, 30, 60, 90, 120, 150]:
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            centroid_x = np.mean(xs)
            centroid_y = np.mean(ys)
            
            for dx in [-0.5, 0.0, 0.5]:
                for dy in [-0.5, 0.0, 0.5]:
                    outer_center = (centroid_x + dx, centroid_y + dy)
                    side = min_outer_side(positions, outer_center, outer_angle)
                    if side < best_side:
                        best_side = side
                        best_inner = [list(p) for p in positions]
                        best_outer_center = outer_center
                        best_outer_angle = outer_angle
    
    # Fine-tune the best configuration with optimization
    if best_inner is not None:
        # Try to optimize the outer center and angle more finely
        def objective(params):
            cx, cy, angle = params
            return min_outer_side(best_inner, (cx, cy), angle)
        
        x0 = [best_outer_center[0], best_outer_center[1], best_outer_angle]
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if result.fun < best_side:
            best_side = result.fun
            best_outer_center = (result.x[0], result.x[1])
            best_outer_angle = result.x[2]
    
    return best_inner, best_outer_center, best_side, best_outer_angle

