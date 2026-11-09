import math
import time

N = 12

def construct():
    # Start time
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Generate the 12 centers
    centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = math.sqrt(3) * (q + r / 2.0)
                y = 1.5 * r
                centers.append((x, y, q, r))
                
    # Sort by distance
    centers.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Select 12
    selected = centers[:12]
    
    # We will try to find the minimum L
    # Scale factor for disjointness
    # Grid distance is sqrt(3). We need > sqrt(3).
    # Let's use 1.0001
    scale = 1.0001
    
    # Vertices of a unit hexagon at angle 0
    # Vertices at (1, 0), (0.5, sqrt(3)/2), ...
    hex_verts = []
    for i in range(6):
        angle = math.pi / 3 * i
        hex_verts.append((math.cos(angle), math.sin(angle)))
        
    def is_inside_outer(x, y, L, outer_angle):
        # Outer hexagon centered at 0,0, side L, rotation outer_angle
        # We check if point (x,y) is inside.
        # Transform point to outer frame
        cos_a = math.cos(outer_angle)
        sin_a = math.sin(outer_angle)
        x_rot = x * cos_a + y * sin_a
        y_rot = -x * sin_a + y * cos_a
        
        # In the rotated frame, outer hexagon has vertices at angles 0, 60...
        # Wait, if angle is rotation of the hexagon, and standard is vertex at 0?
        # Let's assume standard vertex at 0.
        # Check against 3 pairs of parallel sides.
        # Side 1: x = L * sqrt(3)/2 ? No.
        # Vertices: (L, 0), (L/2, L*sqrt(3)/2), (-L/2, L*sqrt(3)/2), ...
        # Lines:
        # 1. x + y/sqrt(3) <= L  (Right side)
        # 2. -x + y/sqrt(3) <= L (Top-Right side? No)
        
        # Let's use the polar radius check.
        # r(phi) = L * sqrt(3) / (2 * cos(phi - 30)) for vertex at 0?
        # If vertex at 0, the side is at 30?
        # Distance to side at 30 is L * cos(30) = L * sqrt(3)/2.
        # r(30) = L * sqrt(3)/2.
        # Formula: r(phi) = (L * sqrt(3)/2) / cos(phi - 30).
        # Valid for phi in [0, 60]?
        # At phi=0, r = L * sqrt(3)/2 / cos(-30) = L * sqrt(3)/2 / (sqrt(3)/2) = L. Correct.
        # At phi=30, r = L * sqrt(3)/2 / 1 = L * sqrt(3)/2. Correct.
        
        phi = math.atan2(y_rot, x_rot)
        # Normalize phi to [0, 60) relative to the nearest vertex/side?
        # The function is periodic with 60.
        # We can map phi to [0, 60] and use the formula for the sector.
        # But the formula depends on which sector.
        # Sector 0: [0, 60]. Vertex at 0 and 60. Side at 30.
        # r(phi) = (L * sqrt(3)/2) / cos(phi - 30).
        
        phi_mod = phi % (math.pi / 3)
        # If phi_mod is in [0, 60], the nearest side is at 30?
        # If phi is close to 0, phi_mod is close to 0. cos(0-30)=cos(-30).
        # If phi is close to 60, phi_mod is close to 60. cos(60-30)=cos(30).
        # So the formula works for [0, 60].
        
        r_bound = (L * math.sqrt(3) / 2.0) / math.cos(phi_mod - math.pi / 6)
        
        r_point = math.sqrt(x_rot**2 + y_rot**2)
        return r_point <= r_bound + 1e-9

    def check_config(L, rotation):
        # rotation is the angle of the inner assembly
        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)
        
        for cx, cy, q, r in selected:
            # Scale and rotate center
            cx_scaled = cx * scale
            cy_scaled = cy * scale
            cx_rot = cx_scaled * cos_r - cy_scaled * sin_r
            cy_rot = cx_scaled * sin_r + cy_scaled * cos_r
            
            # Check vertices of the inner hexagon
            # Inner hexagon is rotated by 'rotation' as well?
            # Yes, we rotate the whole assembly.
            # So the hexagon angle is 'rotation'.
            for vx, vy in hex_verts:
                # Rotate vertex by rotation
                vx_rot = vx * cos_r - vy * sin_r
                vy_rot = vx * sin_r + vy * cos_r
                
                # Add to center
                px = cx_rot + vx_rot
                py = cy_rot + vy_rot
                
                if not is_inside_outer(px, py, L, 0.0):
                    return False
                    
        return True

    # Binary search for L
    # Lower bound: 4.0
    # Upper bound: 4.1
    lo = 4.0
    hi = 4.1
    
    best_L = hi
    best_rot = 0.0
    
    # Try a few rotations to find the best one
    # The assembly has 6-fold symmetry in the ring, but we removed 1 hex.
    # The "gap" breaks symmetry.
    # We should try rotations around 0.
    
    rotations = [0.0, math.pi/12, math.pi/6, -math.pi/12, -math.pi/6]
    
    valid_rotations = []
    for rot in rotations:
        if check_config(hi, rot):
            valid_rotations.append(rot)
            
    if not valid_rotations:
        # If none are valid even at 4.1, return a safe large L
        return [[x*scale, y*scale, 0.0] for x,y,q,r in selected], (0.0, 0.0), 4.1, 0.0

    # For each valid rotation, binary search L
    for rot in valid_rotations:
        lo_r = 4.0
        hi_r = 4.1
        
        for _ in range(20):
            mid = (lo_r + hi_r) / 2.0
            if check_config(mid, rot):
                hi_r = mid
            else:
                lo_r = mid
                
        if hi_r < best_L:
            best_L = hi_r
            best_rot = rot

    # Construct the final result
    inner = []
    cos_r = math.cos(best_rot)
    sin_r = math.sin(best_rot)
    
    for cx, cy, q, r in selected:
        cx_scaled = cx * scale
        cy_scaled = cy * scale
        cx_rot = cx_scaled * cos_r - cy_scaled * sin_r
        cy_rot = cx_scaled * sin_r + cy_scaled * cos_r
        inner.append([cx_rot, cy_rot, best_rot * 180.0 / math.pi])
        
    outer_center = (0.0, 0.0)
    side = best_L
    angle = 0.0
    
    return inner, outer_center, side, angle
