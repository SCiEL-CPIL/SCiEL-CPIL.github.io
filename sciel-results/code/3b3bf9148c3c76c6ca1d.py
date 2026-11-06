"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def design(problem):
    """
    Design a symmetric 32x32 unit cell.
    Uses a central circular Dielectric rod in an Air background.
    Circular symmetry is known to be superior to square symmetry for 
    maximizing band gaps in square lattices due to reduced diffraction 
    along the diagonal directions.
    """
    size = problem["pixels"]
    # Radius in lattice units (0 to 0.5 is half the cell)
    # Optimization for TM 4-5 gap suggests r ~ 0.3
    radius = 0.30
    
    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            # Center of the pixel
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            # Check if inside the circle
            if x * x + y * y < radius * radius:
                row.append(1)  # Dielectric
            else:
                row.append(0)  # Air
        cell.append(row)
    
    return cell
# EVOLVE-BLOCK-END
