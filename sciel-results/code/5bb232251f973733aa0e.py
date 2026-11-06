"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def design(problem):
    """Design a symmetric 32x32 photonic crystal unit cell for maximum TM band gap."""
    size = problem["pixels"]
    
    # Generate a high-quality symmetric pattern.
    # Based on literature, a dielectric rod with radius ~0.3 * a (where a is lattice constant)
    # or an air hole in dielectric background often yields good band gaps.
    # We'll create a central dielectric square/circle hybrid or just a robust circle.
    # A circle of radius 0.32 * size is a good default.
    
    cell = np.zeros((size, size), dtype=int)
    
    # Try a dielectric square instead of a circle.
    # A square rod of side ~0.5-0.6 * a is a common configuration for photonic crystals.
    # We choose side 18 pixels (approx 0.56 * 32) centered.
    side = 18
    start = size // 2 - side // 2
    end = start + side
    
    # Since side is even (18), and size is even (32), the center is between pixels 15 and 16.
    # start = 16 - 9 = 7. end = 7 + 18 = 25.
    # Pixels 7..24 are filled.
    # This is symmetric: 7 and 24 are symmetric around 15.5.
    cell[start:end, start:end] = 1
    
    return cell.tolist()
# EVOLVE-BLOCK-END
