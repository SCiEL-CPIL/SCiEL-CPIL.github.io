"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Hexagonal packing in a 7x3 strip
    # 7 columns, 3 rows (alternating)
    # Rows 0, 2 have 7 circles at x = 1, 2, ..., 7
    # Row 1 has 6 circles at x = 1.5, 2.5, ..., 6.5
    
    # Normalize so that the bounding box has perimeter 4
    # The natural extent in x is from 1 to 7 (width 6)
    # The natural extent in y is from 0 to 2 (height 2)
    # But we need to account for the circle radius r
    
    # Let r be the radius in the normalized coordinate system
    # Circles in row 0: centers at (i, 0) for i = 1..7
    # Circles in row 1: centers at (i+0.5, sqrt(3)/2) for i = 1..6
    # Circles in row 2: centers at (i, sqrt(3)) for i = 1..7
    
    # Bounding box:
    # x_min = 1 - r, x_max = 7 + r => width = 6 + 2r
    # y_min = 0 - r, y_max = sqrt(3) + r => height = sqrt(3) + 2r
    # Perimeter = 2*(width + height) = 2*(6 + 2r + sqrt(3) + 2r) = 2*(6 + sqrt(3) + 4r)
    # Set perimeter = 4:
    # 2*(6 + sqrt(3) + 4r) = 4
    # 6 + sqrt(3) + 4r = 2
    # 4r = 2 - 6 - sqrt(3) = -4 - sqrt(3)
    # This is negative, which means the natural extent is too large.
    
    # Let's rescale. Let the positions be in a coordinate system where we can scale.
    # Let x_pos = [1, 2, 3, 4, 5, 6, 7]
    # Let y_pos = [0, sqrt(3)/2, sqrt(3)]
    # The "unit" radius in this coordinate system is 0.5 (since adjacent circles in same row are distance 1 apart, so radius 0.5)
    # And distance between rows is sqrt(3)/2, and horizontal offset is 0.5, so distance = sqrt(0.5^2 + (sqrt(3)/2)^2) = sqrt(0.25 + 0.75) = 1. So radius 0.5 works.
    
    # So in the natural coordinate system, radius = 0.5.
    # Bounding box width = 7 - 1 + 2*0.5 = 6 + 1 = 7
    # Bounding box height = sqrt(3) - 0 + 2*0.5 = sqrt(3) + 1
    # Perimeter = 2*(7 + sqrt(3) + 1) = 2*(8 + sqrt(3)) = 16 + 2*sqrt(3)
    
    # We need perimeter = 4, so we scale down by factor s = 4 / (16 + 2*sqrt(3))
    s = 4.0 / (16.0 + 2.0 * np.sqrt(3.0))
    
    r = 0.5 * s
    
    cells = []
    # Row 0: y = 0
    for i in range(1, 8):
        cells.append((i * s, 0.0 * s, r))
    # Row 1: y = sqrt(3)/2
    for i in range(1, 7):
        cells.append(((i + 0.5) * s, (np.sqrt(3.0) / 2.0) * s, r))
    # Row 2: y = sqrt(3)
    for i in range(1, 8):
        cells.append((i * s, np.sqrt(3.0) * s, r))
    
    return np.array(cells[:N])
# EVOLVE-BLOCK-END
