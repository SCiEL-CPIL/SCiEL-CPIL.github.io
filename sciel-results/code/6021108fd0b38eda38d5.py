import math
import time

def design(problem):
    """
    Design a 2x2 array of dielectric rods in an air background.
    This structure is known to open larger TM band gaps than a single central rod.
    The rods are centered at (8,8), (8,24), (24,8), (24,24) in a 32x32 cell.
    """
    size = problem["pixels"]
    if size != 32:
        # Fallback for other sizes, though task specifies 32
        size = 32
        
    # Centers of the 4 rods
    # Normalized to 0..1, centers are at 0.25 and 0.75
    # In pixels (0-indexed center of pixel i is i+0.5)
    # We want centers at 1/4 and 3/4 of the cell width.
    # Cell width is 32 pixels.
    # Center of cell is 15.5.
    # 1/4 mark is 8.0 (pixel 8 center is 8.5? No. (8+0.5)/32 = 8.5/32 = 0.265)
    # Let's be precise.
    # Pixel index i covers [i/32, (i+1)/32]. Center is (i+0.5)/32.
    # We want the rod center at 0.25.
    # (i+0.5)/32 = 0.25 => i+0.5 = 8 => i = 7.5.
    # So the center is between pixel 7 and 8.
    # It is better to define the geometry in continuous coordinates and map to pixels.
    
    radius = 0.14  # In units of lattice constant (cell size)
    # Radius in pixels
    r_px = radius * size
    
    centers = [
        (0.25, 0.25),
        (0.25, 0.75),
        (0.75, 0.25),
        (0.75, 0.75)
    ]
    
    cell = []
    for i in range(size):
        row = []
        # y coordinate of pixel center
        y = (i + 0.5) / size
        for j in range(size):
            # x coordinate of pixel center
            x = (j + 0.5) / size
            
            is_dielectric = 0
            for cx, cy in centers:
                dx = x - cx
                dy = y - cy
                if dx*dx + dy*dy < radius*radius:
                    is_dielectric = 1
                    break
            
            row.append(is_dielectric)
        cell.append(row)
        
    return cell
