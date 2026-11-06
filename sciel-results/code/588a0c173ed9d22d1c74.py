import math
import time

def design(problem):
    """
    Design a 2x2 array of dielectric square rods in an air background.
    Square rods often provide larger band gaps than circular rods for the same
    filling fraction in the mid-frequency range due to their different
    scattering properties.
    """
    size = problem["pixels"]
    if size != 32:
        size = 32
        
    # Centers of the 4 rods at 1/4 and 3/4 of the cell
    # Pixel indices for centers:
    # (i+0.5)/32 = 0.25 => i = 7.5 (between 7 and 8)
    # (i+0.5)/32 = 0.75 => i = 23.5 (between 23 and 24)
    
    # We will define squares centered at these points.
    # Let's use a side length of 11 pixels.
    # This gives an area of 121 per rod.
    # Total area = 4 * 121 = 484.
    # Total cell area = 1024.
    # Filling fraction = 484 / 1024 ≈ 0.472.
    
    side = 11
    half_side = side // 2
    
    # Centers in pixel indices (approximate)
    # Center 1: x in [7.5, 8.5], y in [7.5, 8.5]
    # Let's define the square boundaries explicitly to ensure symmetry.
    # Symmetry about 15.5 (center of 32x32).
    # If we place a square from x_start to x_end, it must be symmetric.
    
    # Let's place the squares such that they are centered at 7.5 and 23.5.
    # Square 1 (top-left):
    # x from 7.5 - 5.5 = 2.0 to 7.5 + 5.5 = 13.0
    # Pixels 2 to 12 inclusive?
    # Pixel i covers [i, i+1].
    # If we select pixels 2, 3, ..., 12, that's 11 pixels.
    # Center of these pixels is (2+12)/2 = 7.0. This is not 7.5.
    
    # To be perfectly symmetric about 15.5:
    # If pixel i is ON, pixel 31-i must be ON.
    # Let's try to center the square at 7.5.
    # Pixels 2,3,4,5,6,7,8,9,10,11,12 -> Center 7.
    # Pixels 3,4,5,6,7,8,9,10,11,12,13 -> Center 8.
    # Neither is 7.5.
    
    # However, the symmetry requirement is `cell == cell[::-1]` and `cell == cell.T`.
    # This implies symmetry about the center of the cell (15.5).
    # It does NOT require symmetry about the sub-cell centers (7.5, 23.5) individually,
    # but the 2x2 pattern combined with global symmetry enforces it.
    
    # Let's pick a square size that allows perfect centering on the grid or close enough.
    # If we use odd side length 11, we can't center on .5.
    # If we use even side length 10, we can center on .5.
    # Pixels 3..12 (10 pixels). Center (3+12)/2 = 7.5.
    # Pixels 19..28 (10 pixels). Center (19+28)/2 = 23.5.
    
    # Let's use side = 10.
    # Area = 100. Total = 400. FF = 400/1024 ≈ 0.39.
    # This is a bit low. Let's try side = 12?
    # Pixels 2..13 (12 pixels). Center 7.5.
    # Area = 144. Total = 576. FF = 576/1024 = 0.5625.
    
    # Let's try side = 11 but shift to be symmetric?
    # If we use pixels 2..12 (center 7) and 19..29 (center 24)?
    # 31-2 = 29. 31-12 = 19.
    # So pixels 2-12 and 19-29 are symmetric.
    # Center of 2-12 is 7. Center of 19-29 is 24.
    # This is symmetric about 15.5.
    # Distance from center 15.5 to 7 is 8.5. To 24 is 8.5.
    # This works.
    
    # Let's use side = 11, pixels 2-12.
    side = 11
    # Define the ranges
    # x1_start = 2, x1_end = 12
    # x2_start = 19, x2_end = 29
    # Same for y.
    
    x_starts = [2, 19]
    y_starts = [2, 19]
    
    cell = [[0 for _ in range(size)] for _ in range(size)]
    
    for sx in x_starts:
        for sy in y_starts:
            for i in range(sy, sy + side):
                for j in range(sx, sx + side):
                    if 0 <= i < size and 0 <= j < size:
                        cell[i][j] = 1
                        
    return cell
