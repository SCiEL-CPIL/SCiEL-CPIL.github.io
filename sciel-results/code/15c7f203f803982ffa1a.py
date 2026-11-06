"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def design(problem):
    """
    A 2x2 array of dielectric rods centered in the four quadrants of the unit cell.
    This structure is known to enhance Bragg scattering for higher-order TM band gaps
    compared to a single central rod.
    """
    size = problem["pixels"]
    cell = [[0 for _ in range(size)] for _ in range(size)]
    
    # Centers of the 4 rods, one in each quadrant
    # Using 0.5 offset for pixel centers
    # Quadrants: (0,0), (0,1), (1,0), (1,1) in a 2x2 supercell sense
    # 32x32 cell -> each quadrant is 16x16
    # Centers at 8.5 and 23.5
    
    centers = [
        (8.5, 8.5),
        (8.5, 23.5),
        (23.5, 8.5),
        (23.5, 23.5)
    ]
    
    # Radius of the rods. 
    # A radius of 4.5 gives a diameter of 9 pixels.
    # This is a reasonable starting point for high-contrast gaps.
    radius = 4.5
    r_sq = radius * radius
    
    for i in range(size):
        y = (i + 0.5) - size / 2.0
        for j in range(size):
            x = (j + 0.5) - size / 2.0
            # Check if pixel is inside any of the 4 rods
            # Note: x, y are centered at 0,0. 
            # Centers in centered coords:
            # (8.5, 8.5) -> (8.5 - 16, 8.5 - 16) = (-7.5, -7.5)
            # (8.5, 23.5) -> (-7.5, 7.5)
            # (23.5, 8.5) -> (7.5, -7.5)
            # (23.5, 23.5) -> (7.5, 7.5)
            
            # It's easier to just use the absolute coordinates i, j and compare to centers
            in_rod = False
            for cx, cy in centers:
                dx = (i + 0.5) - cx
                dy = (j + 0.5) - cy
                if dx*dx + dy*dy < r_sq:
                    in_rod = True
                    break
            cell[i][j] = 1 if in_rod else 0
            
    return cell
# EVOLVE-BLOCK-END
