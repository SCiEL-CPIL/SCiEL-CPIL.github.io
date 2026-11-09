"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def design(problem):
    """
    Designs a 32x32 unit cell with a centered cross-shaped dielectric block.
    Cross shapes are known to enhance higher-order band gaps in square-lattice
    photonic crystals by providing strong coupling in both x and y directions
    while maintaining 4-fold symmetry.
    """
    size = problem["pixels"]
    
    # Parameters for the cross shape
    # Arm width and length (in units of the lattice constant, i.e., cell size)
    arm_width = 0.25
    arm_length = 0.75
    
    cell = []
    for i in range(size):
        # Coordinate in [0, 1)
        y = (i + 0.5) / size
        row = []
        for j in range(size):
            x = (j + 0.5) / size
            # Center is at (0.5, 0.5)
            dx = abs(x - 0.5)
            dy = abs(y - 0.5)
            
            # Cross condition:
            # (x is within arm_width/2 of center AND y is within arm_length/2 of center)
            # OR
            # (y is within arm_width/2 of center AND x is within arm_length/2 of center)
            in_horizontal_arm = (dx < arm_width / 2.0) and (dy < arm_length / 2.0)
            in_vertical_arm = (dy < arm_width / 2.0) and (dx < arm_length / 2.0)
            
            if in_horizontal_arm or in_vertical_arm:
                row.append(1)
            else:
                row.append(0)
        cell.append(row)
    return cell
# EVOLVE-BLOCK-END
