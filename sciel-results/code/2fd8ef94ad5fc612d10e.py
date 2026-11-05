"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def design(problem):
    """
    Design a super-ellipse dielectric rod to maximize the TM band gap between bands 4 and 5.
    Uses a super-ellipse shape with exponent n=4 to leverage the square lattice symmetry.
    """
    size = problem["pixels"]
    # Parameters for the super-ellipse
    # n=4 gives a shape between a circle and a square, often optimal for square lattices
    n_exp = 4.0
    # Radius in lattice units (0.0 to 0.5)
    # Seed was 0.2. Increasing size usually increases the gap until a point.
    # 0.35 is a common optimal radius for TM gaps in square lattices.
    radius = 0.35
    
    cell = []
    for i in range(size):
        row = []
        x = (i + 0.5) / size - 0.5
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            # Super-ellipse equation: |x/a|^n + |y/a|^n < 1
            # Using absolute values to handle negative coordinates
            val = abs(x / radius) ** n_exp + abs(y / radius) ** n_exp
            if val < 1.0:
                row.append(1)
            else:
                row.append(0)
        cell.append(row)
    return cell
# EVOLVE-BLOCK-END
