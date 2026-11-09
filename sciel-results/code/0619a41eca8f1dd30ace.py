"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell for a 2D square-lattice photonic crystal
    that maximizes the gap between TM bands 4 and 5.
    
    Strategy: Use a dielectric background with a central air square (hole).
    This "inverse" topology (holes in dielectric vs rods in air) often yields
    different band structures. For TM polarization, confinement in the
    high-epsilon background can be effective. The square hole (Chebyshev distance)
    aligns with the lattice axes, enhancing scattering along Gamma-M and Gamma-X.
    
    The cell must be symmetric under:
    - 180-degree rotation
    - Transpose symmetry
    
    We design a centered square air hole in a dielectric medium.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    
    cell = [[1] * size for _ in range(size)]  # Start with all dielectric
    
    # Center of the 32x32 grid
    center = (size - 1) / 2.0
    
    # Parameters for the square air hole
    # Using Chebyshev distance: max(|x - cx|, |y - cy|)
    # This creates a square aligned with the axes.
    r_hole = 0.25 * size  # ~8 pixels radius, so 16x16 hole
    
    for i in range(size):
        for j in range(size):
            # Chebyshev distance from center
            d = max(abs(i - center), abs(j - center))
            
            if d < r_hole:
                cell[i][j] = 0  # Air hole
            else:
                cell[i][j] = 1  # Dielectric background
    
    return cell
# EVOLVE-BLOCK-END
