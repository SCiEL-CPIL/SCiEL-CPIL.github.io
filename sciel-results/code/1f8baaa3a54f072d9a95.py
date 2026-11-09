"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell for a 2D square-lattice photonic crystal
    that maximizes the gap between TM bands 4 and 5.
    
    Strategy: Use a dielectric rod with a hole in the center (annular/donut shape).
    This is a common design that can open larger band gaps than a solid rod
    for certain band indices, particularly higher-order gaps.
    
    The cell must be symmetric under:
    - 180-degree rotation (cell == cell[::-1] row-wise and column-wise)
    - Transpose symmetry (cell == cell.T)
    
    We design a centered annulus (ring) of dielectric material.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    
    # We'll use a donut/annulus shape: dielectric between r_inner and r_outer
    # Center is at (size/2, size/2)
    center = (size - 1) / 2.0
    
    cell = [[0] * size for _ in range(size)]
    
    # Try an annular shape
    # For a 32x32 grid, the center is at 15.5
    # Let's define radii in pixel units
    # r_inner and r_outer determine the ring thickness
    
    # Based on literature, for higher-order TM gaps in square lattices,
    # an annular rod can be effective. Let's try r_inner ~ 0.15*a, r_outer ~ 0.35*a
    # where a is the lattice constant = size pixels
    
    r_inner = 0.12 * size  # ~3.84 pixels
    r_outer = 0.38 * size  # ~12.16 pixels
    
    for i in range(size):
        for j in range(size):
            # Distance from center in pixels
            dx = i - center
            dy = j - center
            r = np.sqrt(dx * dx + dy * dy)
            
            if r_inner <= r <= r_outer:
                cell[i][j] = 1
            else:
                cell[i][j] = 0
    
    return cell
# EVOLVE-BLOCK-END
