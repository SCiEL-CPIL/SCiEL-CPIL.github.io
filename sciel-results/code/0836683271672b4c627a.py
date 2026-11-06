"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def design(problem):
    """
    Design a 32x32 unit cell with four dielectric rods at (±0.25, ±0.25) 
    to enhance second-order Fourier components for the 4-5 TM band gap.
    """
    start_time = time.time()
    deadline = start_time + 220.0  # 10s safety margin
    
    size = problem.get("pixels", 32)
    eps_high = problem.get("eps_high", 12.0)
    
    # Parameters for the four-rod design
    # Centers at (±0.25, ±0.25) in normalized coordinates [-0.5, 0.5]
    # Radius chosen to provide sufficient volume and strong 2nd order components
    radius = 0.15
    centers = [
        (-0.25, -0.25),
        ( 0.25, -0.25),
        (-0.25,  0.25),
        ( 0.25,  0.25)
    ]
    
    cell = [[0 for _ in range(size)] for _ in range(size)]
    
    # Generate the pixel map
    for i in range(size):
        # y coordinate in [-0.5, 0.5]
        y = (i + 0.5) / size - 0.5
        for j in range(size):
            # x coordinate in [-0.5, 0.5]
            x = (j + 0.5) / size - 0.5
            
            # Check if this pixel is within any of the four rods
            inside = False
            for cx, cy in centers:
                dist_sq = (x - cx)**2 + (y - cy)**2
                if dist_sq < radius**2:
                    inside = True
                    break
            
            if inside:
                cell[i][j] = 1
            else:
                cell[i][j] = 0
                
    # Ensure strict symmetry as per contract
    # The generation above is symmetric by construction, but let's enforce it 
    # to be absolutely safe against floating point edge cases or logic errors.
    # 1. Vertical flip: cell[i][j] == cell[size-1-i][j]
    # 2. Horizontal flip: cell[i][j] == cell[i][size-1-j]
    # 3. Transpose: cell[i][j] == cell[j][i]
    
    # The four-rod pattern at (±0.25, ±0.25) is inherently symmetric under 
    # x->-x, y->-y, and x<->y.
    # Pixel (i,j) corresponds to (x,y).
    # Flip y: y' = -y. (x,y) -> (x,-y).
    # Flip x: x' = -x. (x,y) -> (-x,y).
    # Transpose: (x,y) -> (y,x).
    
    # Since the set of centers {(-0.25,-0.25), (0.25,-0.25), (-0.25,0.25), (0.25,0.25)}
    # is invariant under these transformations, the generated map should be symmetric.
    # However, pixel centers are at (i+0.5)/N - 0.5.
    # i=0 -> -0.5 + 0.5/32
    # i=31 -> 0.5 - 0.5/32
    # The grid is symmetric. The function f(x,y) = sum of indicators is symmetric.
    # So cell[i][j] should equal cell[31-i][j], etc.
    
    # No explicit enforcement loop needed if logic is correct, 
    # but we can return the cell directly.
    
    return cell

# EVOLVE-BLOCK-END
