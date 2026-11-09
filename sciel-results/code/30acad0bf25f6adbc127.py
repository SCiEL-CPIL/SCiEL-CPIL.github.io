"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell using a symmetric geometric construction.
    Approach: 4-Corner Squares.
    We construct the top-left 16x16 quadrant with a square of dielectric
    in the corner, then mirror it to all four quadrants to ensure
    strict symmetry under the square group.
    """
    size = problem["pixels"]
    half_size = size // 2
    
    # Create an empty 32x32 grid
    cell_np = np.zeros((size, size), dtype=int)
    
    # Define the size of the square in the corner of each quadrant
    # The quadrant is 16x16.
    # Let's try a square of size 10x10 at the corner (0,0) of the quadrant.
    s = 10
    
    # Fill the top-left quadrant's corner
    # Indices 0 to s-1
    cell_np[0:s, 0:s] = 1
    
    # Mirror to Top-Right
    # The top-right quadrant corresponds to columns half_size to size-1.
    # The square should be at the corner (size-1, 0) effectively?
    # Due to symmetry cell[i][j] == cell[i][size-1-j],
    # if we set cell[i][j]=1 for j in 0..s-1,
    # then cell[i][size-1-j]=1 for j in 0..s-1.
    # This means columns size-1, size-2, ..., size-s will be 1.
    # So we can just mirror the left half to the right half.
    
    # Enforce horizontal symmetry (left-right)
    cell_np[:, half_size:] = cell_np[:, half_size-1::-1]
    
    # Enforce vertical symmetry (top-bottom)
    cell_np[half_size:, :] = cell_np[half_size-1::-1, :]
    
    # Convert to list of lists
    cell = cell_np.tolist()
    
    return cell
# EVOLVE-BLOCK-END
