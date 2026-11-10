"""
Designs a 32x32 photonic crystal unit cell to maximize the TM band gap between bands 4 and 5.
Uses a dielectric background (epsilon=12) with a central air hole and four symmetric side air holes.
This "rosette" structure is known to open significant higher-order gaps in square lattices.
"""
import math

def design(problem):
    size = problem["pixels"]
    # Initialize with dielectric (1)
    cell = [[1 for _ in range(size)] for _ in range(size)]
    
    # Center of the cell
    cx = (size - 1) / 2.0
    cy = (size - 1) / 2.0
    
    # Parameters for the 5-hole structure (in units of cell size)
    # These values are chosen based on typical optimal geometries for mid-band gaps in square lattices
    r_central = 0.18 * size
    r_side = 0.13 * size
    d_side = 0.38 * size
    
    # Define hole centers
    holes = [
        (cx, cy, r_central),
        (cx - d_side, cy, r_side),
        (cx + d_side, cy, r_side),
        (cx, cy - d_side, r_side),
        (cx, cy + d_side, r_side)
    ]
    
    # Carve out the holes (set to 0 for air)
    for i in range(size):
        for j in range(size):
            # Pixel center coordinates
            px = j + 0.5
            py = i + 0.5
            
            for hx, hy, hr in holes:
                dist_sq = (px - hx)**2 + (py - hy)**2
                if dist_sq < hr**2:
                    cell[i][j] = 0
                    break
    
    return cell
