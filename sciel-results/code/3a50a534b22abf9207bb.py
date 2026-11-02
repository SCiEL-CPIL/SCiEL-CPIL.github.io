"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Layout: 5 rows of hexagonal packing
    # Row counts: 5, 4, 5, 4, 3
    row_counts = [5, 4, 5, 4, 3]
    
    centers = []
    row_y = 0.0
    for i, count in enumerate(row_counts):
        # Hexagonal offset: even rows start at 0, odd rows start at 1 (in units of radius)
        # This assumes a standard hexagonal lattice where horizontal spacing is 2r
        # and vertical spacing is sqrt(3)r.
        if i % 2 == 0:
            start_x = 0.0
        else:
            start_x = 1.0
            
        for j in range(count):
            x = start_x + 2.0 * j
            y = row_y
            centers.append([x, y])
            
        row_y += np.sqrt(3.0)
        
    centers = np.array(centers)
    
    # Calculate span of centers
    x_min, y_min = centers.min(axis=0)
    x_max, y_max = centers.max(axis=0)
    
    span_x = x_max - x_min
    span_y = y_max - y_min
    
    # The bounding box of the circles (radius r) has:
    # Width W = span_x + 2r
    # Height H = span_y + 2r
    # Constraint: W + H <= 2  =>  span_x + span_y + 4r <= 2
    # Maximize sum of radii = N * r
    # So we maximize r subject to 4r <= 2 - (span_x + span_y)
    # r_max = (2 - (span_x + span_y)) / 4
    
    # Note: span_x and span_y are in units of r=1.
    # So if we scale the whole configuration by s, the new radius is R = s.
    # The new span_x becomes s * span_x.
    # Constraint: s * span_x + 2s + s * span_y + 2s <= 2
    # s * (span_x + span_y + 4) <= 2
    # s = 2 / (span_x + span_y + 4)
    
    s = 2.0 / (span_x + span_y + 4.0)
    
    R = s
    final_centers = centers * s
    
    # Construct output array: x, y, r
    result = np.column_stack((final_centers[:, 0], final_centers[:, 1], np.full(N, R)))
    
    return result
# EVOLVE-BLOCK-END
