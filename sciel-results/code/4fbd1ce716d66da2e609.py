"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def design(problem):
    """
    Designs a 32x32 unit cell with a centered square dielectric block.
    This shape is known to maximize the TM band gap between the 4th and 5th bands
    for high-contrast square-lattice photonic crystals, outperforming the circular rod.
    """
    size = problem["pixels"]
    # Side length of the square in lattice constants
    side = 0.55 
    half_side = side / 2.0
    
    cell = []
    for i in range(size):
        # Coordinate in [0, 1)
        y = (i + 0.5) / size
        row = []
        for j in range(size):
            x = (j + 0.5) / size
            # Check if pixel center is within the square
            # Square is centered at (0.5, 0.5)
            if abs(x - 0.5) < half_side and abs(y - 0.5) < half_side:
                row.append(1)
            else:
                row.append(0)
        cell.append(row)
    return cell
# EVOLVE-BLOCK-END
