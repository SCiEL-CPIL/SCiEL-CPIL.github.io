import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Define 11 hexagon centers
    # Using a 13-hex cluster (side 2) and removing 2 corners to potentially reduce span
    # Or just use the seed's 11-hex cluster.
    # Let's use the 13-hex cluster and remove 2 opposite corners to keep symmetry and reduce max radius.
    # 13-hex cluster centers:
    # (0,0)
    # Ring 1: distance sqrt(3), angles 0, 60, ...
    # Ring 2: distance 2*sqrt(3), angles 0, 60, ... (6 hexes)
    # Total 13.
    # Remove 2 opposite from Ring 2, e.g., at 0 and 180.
    # Remaining Ring 2: 60, 120, 180 (removed), 240, 300 (removed)? No, 0 and 180 removed.
    # Remaining: 60, 120, 240, 300. (4 hexes).
    # This leaves the cluster with "flat" sides on left and right?
    
    centers = []
    centers.append((0.0, 0.0))
    
    # Ring 1
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))
        
    # Ring 2 (4 hexes, skipping 0 and 180 degrees)
    for k in [1, 2, 4, 5]: # Angles 60, 120, 240, 300
        angle = np.deg2rad(k * 60)
        r = 2 * np.sqrt(3)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))
        
    # Center the cluster
    min_x = min(c[0] for c in centers)
    max_x = max(c[0] for c in centers)
    min_y = min(c[1] for c in centers)
    max_y = max(c[1] for c in centers)
    
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    
    shifted_centers = [(x - cx, y - cy) for x, y in centers]
    
    # Function to check if a point is inside a regular hexagon of side S, angle 0 (flat top? or pointy?)
    # Let's assume angle 0 means "standard" orientation.
    # For Minkowski difference, we need the inner hexagon (side S-1) to contain the centers.
    # The inner hexagon (for centers) has side S_center = S - 1.
    # We align it with angle 0.
    
    def point_in_hexagon_center(px, py, side_center):
        # side_center is the side length of the hexagon that must contain the point.
        # If side_center < 0, impossible.
        if side_center < 0:
            return False
            
        apothem = side_center * np.sqrt(3) / 2.0
        
        # Check 6 half-planes.
        # Normals at 0, 60, 120, 180, 240, 300?
        # If angle 0 is "flat top" (edges horizontal), normals are vertical?
        # Let's use the property: max(|x|, |y|, |x+y|/sqrt(3)...) ?
        # Simpler: Use the distance to edges.
        # For a hexagon with flat top (vertices at 30, 90...), the width is 2*S, height is sqrt(3)*S.
        # Normals are at 0, 60, 120, 180, 240, 300?
        # If vertices are at 30, 90, 150, 210, 270, 330.
        # Edge 0: between 330 and 30. Midpoint 0. Normal 0.
        # So normals are at 0, 60, 120, 180, 240, 300.
        
        for k in range(6):
            normal_angle = np.deg2rad(k * 60)
            nx = np.cos(normal_angle)
            ny = np.sin(normal_angle)
            proj = px * nx + py * ny
            if proj > apothem + 1e-8:
                return False
            if proj < -apothem - 1e-8:
                return False
        return True

    def is_valid_configuration(side):
        side_center = side - 1.0
        if side_center < 0:
            return False
        for cx, cy in shifted_centers:
            if not point_in_hexagon_center(cx, cy, side_center):
                return False
        return True

    # Binary search
    # Lower bound: 1.0 (for 1 hex)
    # Upper bound: Start with 5.0 (for 13 hexes, S=5 is safe)
    s_lower = 1.0
    s_upper = 5.0
    
    # Ensure s_upper is valid
    while not is_valid_configuration(s_upper):
        s_upper *= 1.1
        if time.time() > deadline:
            break

    if time.time() > deadline:
        return [(x, y, 0.0) for x, y in shifted_centers], (0.0, 0.0), s_upper, 0.0

    for _ in range(50):
        if time.time() > deadline:
            break
        mid = (s_lower + s_upper) / 2.0
        if is_valid_configuration(mid):
            s_upper = mid
        else:
            s_lower = mid
            
    side = s_upper

    inner = []
    for x, y in shifted_centers:
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), side, 0.0
