"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell with four dielectric rods in the quadrants.
    This introduces 2x2 sub-periodicity which is known to enhance the TM band gap.
    """
    size = problem["pixels"]
    cell = [[0 for _ in range(size)] for _ in range(size)]
    
    # Centers of the four 16x16 quadrants
    # Using pixel centers (i+0.5, j+0.5)
    # Quadrant centers at 7.5, 23.5
    centers = [
        (7.5, 7.5),
        (23.5, 7.5),
        (7.5, 23.5),
        (23.5, 23.5)
    ]
    
    # Radius tuned for optimal gap in 2x2 supercell configuration
    # Lattice constant of substructure is 16 pixels.
    # Radius 5.0 corresponds to diameter 10, which is ~62% of sub-lattice constant.
    # This is a common high-fill-fraction regime for TM gaps.
    radius = 5.0
    
    for i in range(size):
        for j in range(size):
            x = i + 0.5
            y = j + 0.5
            for cx, cy in centers:
                if (x - cx)**2 + (y - cy)**2 < radius**2:
                    cell[i][j] = 1
                    break
    
    return cell
# EVOLVE-BLOCK-END
