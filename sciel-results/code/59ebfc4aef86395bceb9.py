"""
Rewritten program for photonic crystal design.
Approach: Inverse Lattice (Dielectric Background) with Cross-Shaped Air Holes.
This explores a different design space from the seed (Direct Lattice/Rods).
"""
# EVOLVE-BLOCK-START
import numpy as np

def design(problem):
    """
    Design a symmetric 32x32 unit cell of a square-lattice photonic crystal
    that maximizes the TM band gap between bands 4 and 5.
    
    Strategy: Inverse Lattice with Cross-Shaped Air Holes.
    - Background is dielectric (epsilon=12, value 1).
    - Central feature is an air cross (epsilon=1, value 0).
    - This geometry is known to support strong TM modes due to the high-index
      background and the specific field confinement in the cross arms.
    - The cross shape breaks the circular symmetry, which can lift degeneracies
      and open gaps between specific bands (like 4 and 5) that might be degenerate
      or close in a circular hole structure.
    
    Parameters optimized for 32x32 grid:
    - Cross Total Width (W): 20 pixels (extent from center)
    - Cross Arm Thickness (T): 8 pixels (width of the arms)
    """
    size = problem["pixels"]
    
    # Initialize with dielectric (1)
    cell = [[1 for _ in range(size)] for _ in range(size)]
    
    # Define cross parameters
    # For a 32x32 grid, center is at 15.5.
    # We want the cross to be symmetric.
    
    # Total extent of the cross (half-width of the bounding box)
    # If W_total is the full width, half is W_total/2.
    # Let's use pixel indices.
    # Center is between 15 and 16.
    
    # Let's try a cross with arm thickness 8 and total span 20.
    # Arm thickness 8 means it covers 4 pixels on each side of the center line.
    # Center lines are between 15-16.
    # Pixels 12, 13, 14, 15, 16, 17, 18, 19 are within 4 pixels of 15.5.
    # |12 - 15.5| = 3.5 < 4.0
    # |19 - 15.5| = 3.5 < 4.0
    # So arm pixels are 12..19.
    
    # Total span 20 means it covers 10 pixels on each side of the center line.
    # |i - 15.5| < 10.0
    # i=6: |6-15.5|=9.5 < 10.0
    # i=25: |25-15.5|=9.5 < 10.0
    # So span pixels are 6..25.
    
    arm_half_thick = 4.0  # Half of the arm thickness in pixels
    span_half_width = 10.0 # Half of the total span in pixels
    
    for i in range(size):
        for j in range(size):
            dx = abs(i - (size - 1) / 2.0)
            dy = abs(j - (size - 1) / 2.0)
            
            # Cross condition:
            # Inside horizontal arm: dx < arm_half_thick AND dy < span_half_width
            # Inside vertical arm:   dy < arm_half_thick AND dx < span_half_width
            
            in_h_arm = (dx < arm_half_thick) and (dy < span_half_width)
            in_v_arm = (dy < arm_half_thick) and (dx < span_half_width)
            
            if in_h_arm or in_v_arm:
                cell[i][j] = 0  # Air
            else:
                cell[i][j] = 1  # Dielectric
                
    return cell
# EVOLVE-BLOCK-END
