"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def design(problem):
    """
    Design a 32x32 unit cell with a centered square dielectric rod.
    The square shape is preferred over circular for TM gaps in square lattices 
    due to better coupling of higher-order modes.
    We use a side length of approximately 0.55 lattice constants (17-18 pixels).
    """
    size = problem["pixels"]
    # Time guard
    start_time = time.time()
    
    # Center of the cell
    # Pixels are indexed 0 to 31.
    # Center is at 15.5.
    # A square of half-width H (in pixels) covers indices center-H to center+H.
    
    # Strategy: Try a few candidate half-widths and shapes, but return the best static one.
    # Based on literature, a square rod with side ~0.5-0.6 a is good for TM gaps.
    # Let's use a half-width of 0.28 * size (approx 9 pixels).
    # Side length = 2 * 9 = 18 pixels. 18/32 = 0.5625.
    
    # We will construct a square.
    # To ensure symmetry: cell[i][j] == cell[size-1-i][size-1-j] etc.
    
    half_width_px = 9.0
    
    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            # Coordinates relative to center (15.5, 15.5)
            x = i - (size - 1) / 2.0
            y = j - (size - 1) / 2.0
            
            # Check if inside the square
            if abs(x) <= half_width_px and abs(y) <= half_width_px:
                row.append(1)
            else:
                row.append(0)
        cell.append(row)
        
    # Verify symmetry and return
    return cell
# EVOLVE-BLOCK-END
