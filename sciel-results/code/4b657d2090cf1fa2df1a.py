"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def design(problem):
    """
    Design a symmetric 32x32 unit cell.
    Strategy: Use a dielectric background (epsilon=12) with a central air hole (epsilon=1).
    This inverted structure typically yields larger band gaps for higher TM bands
    compared to the air-background rod structure used in the seed.
    """
    size = problem["pixels"]
    cell = []
    
    # Parameters for the central air hole
    # Radius in units of the lattice constant (cell size = 1)
    radius = 0.35
    
    # We iterate through the pixels
    # The cell is centered at (0.5, 0.5) in normalized coordinates
    # Pixel (i, j) center is at (x, y) = ((i + 0.5) / size - 0.5, (j + 0.5) / size - 0.5)
    
    for i in range(size):
        row = []
        x = (i + 0.5) / size - 0.5
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            dist_sq = x * x + y * y
            # If inside the hole, it is air (0). Otherwise, it is dielectric (1).
            if dist_sq < radius * radius:
                row.append(0)
            else:
                row.append(1)
        cell.append(row)
        
    return cell
# EVOLVE-BLOCK-END
