"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 pixel unit cell with a dielectric background (eps=12) 
    and a central circular air hole (eps=1).
    This geometry is chosen to maximize the TM band gap between bands 4 and 5.
    """
    size = problem["pixels"]
    # Initialize with dielectric (1)
    cell = [[1 for _ in range(size)] for _ in range(size)]
    
    # Optimal radius for air hole in dielectric background for TM polarization
    # Normalized to lattice constant a=1.
    # Typical optimal range for square lattice TM gaps is r/a ~ 0.3-0.4.
    radius = 0.35
    r_sq = radius * radius
    
    for i in range(size):
        # Coordinate x in [-0.5, 0.5]
        x = (i + 0.5) / size - 0.5
        x_sq = x * x
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            # If inside the circle, it's air (0)
            if x_sq + y * y < r_sq:
                cell[i][j] = 0
                
    return cell
# EVOLVE-BLOCK-END
