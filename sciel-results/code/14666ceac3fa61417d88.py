import numpy as np

def construct():
    # Hexagonal packing for 21 circles
    # Rows: 4, 5, 4, 5, 3
    # y-coordinates: 0, sqrt(3), 2*sqrt(3), 3*sqrt(3), 4*sqrt(3)
    # x-coordinates centered around 0
    
    sqrt3 = np.sqrt(3)
    
    # Row 0 (4 circles): y=0
    # To minimize width, center them.
    # x: -3, -1, 1, 3
    row0_x = np.array([-3, -1, 1, 3])
    row0_y = np.zeros(4)
    
    # Row 1 (5 circles): y=sqrt(3)
    # Offset by 1 unit relative to row 0?
    # Row 0 has even count, Row 1 odd.
    # If Row 0 is -3,-1,1,3, Row 1 should be -4,-2,0,2,4 to interlock.
    row1_x = np.array([-4, -2, 0, 2, 4])
    row1_y = np.full(5, sqrt3)
    
    # Row 2 (4 circles): y=2*sqrt(3)
    # Align with Row 0
    row2_x = np.array([-3, -1, 1, 3])
    row2_y = np.full(4, 2*sqrt3)
    
    # Row 3 (5 circles): y=3*sqrt(3)
    # Align with Row 1
    row3_x = np.array([-4, -2, 0, 2, 4])
    row3_y = np.full(5, 3*sqrt3)
    
    # Row 4 (3 circles): y=4*sqrt(3)
    # Offset from Row 3.
    # Row 3 is -4,-2,0,2,4.
    # Row 4 should be -3,-1,1 or -1,1,3.
    # Let's choose -2, 0, 2? No, that overlaps with Row 3's -2,0,2 vertically?
    # Row 3 x: -2, 0, 2. Row 4 x: -2, 0, 2.
    # Dist: sqrt(0^2 + sqrt(3)^2) = sqrt(3) < 2. Overlap.
    # So Row 4 must be offset.
    # Options: -3, -1, 1 or -1, 1, 3.
    # Let's pick -2, 0, 2 is WRONG.
    # Pick -3, -1, 1?
    # Check overlap with Row 3 (-4, -2, 0, 2, 4).
    # -3 vs -4, -2 (dist 1). OK.
    # -1 vs -2, 0 (dist 1). OK.
    # 1 vs 0, 2 (dist 1). OK.
    row4_x = np.array([-3, -1, 1])
    row4_y = np.full(3, 4*sqrt3)
    
    # Combine
    xs = np.concatenate([row0_x, row1_x, row2_x, row3_x, row4_x])
    ys = np.concatenate([row0_y, row1_y, row2_y, row3_y, row4_y])
    
    # Center the configuration
    xs -= np.mean(xs)
    ys -= np.mean(ys)
    
    # Calculate bounding box dimensions in units of r=1
    min_x = np.min(xs)
    max_x = np.max(xs)
    min_y = np.min(ys)
    max_y = np.max(ys)
    
    width_c = max_x - min_x
    height_c = max_y - min_y
    
    # Physical dimensions with radius r
    # W = width_c + 2r
    # H = height_c + 2r
    # 2(W+H) = 4 => W+H = 2
    # width_c + 2r + height_c + 2r = 2
    # 4r = 2 - (width_c + height_c)
    
    r = (2 - (width_c + height_c)) / 4.0
    
    if r <= 0:
        # Fallback to grid if hexagonal is too large (should not happen)
        side = 5
        r = (2 - (2*(side-1) + 2*(side-1))) / 4.0 # Grid span 8+8=16?
        # Grid 5x5 span 8. 2(8+8+4r)=4 => 32+8r=4 impossible.
        # Grid 5x5: W=10, H=10. 2(20+4r)=4 => 40+8r=4.
        # Wait.
        # W = 8 + 2r. H = 8 + 2r.
        # 2(16 + 4r) = 4 => 32 + 8r = 4.
        # r is negative.
        # The grid 5x5 does NOT fit in perimeter 4?
        # Perimeter 4 => Square side 1.
        # 5x5 grid in 1x1 square.
        # r = 1/10 = 0.1.
        # W = 8*0.1 + 0.2 = 1.0. H = 1.0.
        # Perimeter 4.
        # My formula: 4r = 2 - 16 = -14.
        # Error in formula:
        # W = width_c * r + 2r ? No.
        # width_c is span in units of r=1.
        # If we scale by r, width_c becomes width_c * r.
        # W = width_c * r + 2r = r(width_c + 2).
        # H = r(height_c + 2).
        # 2(r(width_c+2) + r(height_c+2)) = 4.
        # 2r(width_c + height_c + 4) = 4.
        # r = 2 / (width_c + height_c + 4).
        
        r = 2.0 / (width_c + height_c + 4.0)
        
        # Recalculate xs, ys scaled by r
        xs = xs * r
        ys = ys * r
        
        # Center in the rectangle
        # W = r(width_c + 2)
        # H = r(height_c + 2)
        # Center x = 0?
        # The bounding box of centers is [min_x*r, max_x*r].
        # The circle centers should be shifted so that the bounding box of circles is [0, W] x [0, H].
        # Min circle x = min_x*r - r.
        # Max circle x = max_x*r + r.
        # We want Min circle x = 0.
        # Shift = - (min_x*r - r) = r - min_x*r = r(1 - min_x).
        
        shift_x = r * (1 - min_x)
        shift_y = r * (1 - min_y)
        
        xs = xs + shift_x
        ys = ys + shift_y
        
    else:
        # Correct formula
        r = 2.0 / (width_c + height_c + 4.0)
        xs = xs * r
        ys = ys * r
        shift_x = r * (1 - min_x)
        shift_y = r * (1 - min_y)
        xs = xs + shift_x
        ys = ys + shift_y

    # Create result array
    result = np.column_stack((xs, ys, np.full_like(xs, r)))
    
    return result
