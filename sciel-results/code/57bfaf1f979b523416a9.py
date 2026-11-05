"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell with four dielectric features in the quadrants.
    Uses a rounded-square (super-ellipse) shape to modify the Fourier spectrum
    compared to a pure circle, potentially enhancing the TM band gap between
    bands 4 and 5.
    """
    size = problem["pixels"]
    half = size // 2
    
    # Parameters for the feature in the top-left quadrant
    # Center of the 16x16 quadrant (indices 0-15) is at 7.5, 7.5
    cx = 7.5
    cy = 7.5
    # Use a super-ellipse with exponent n=4 (rounded square) and size ~5
    a = 5.0  # half-width
    n = 4.0  # super-ellipse exponent
    
    # Initialize the top-left quadrant
    quadrant = [[0 for _ in range(half)] for _ in range(half)]
    
    # Fill the quadrant with super-ellipse shape
    for i in range(half):
        for j in range(half):
            x = i + 0.5
            y = j + 0.5
            dx = abs(x - cx) / a
            dy = abs(y - cy) / a
            # Super-ellipse: |dx|^n + |dy|^n <= 1
            if dx**n + dy**n < 1.0:
                quadrant[i][j] = 1
    
    # Construct the full 32x32 cell by mirroring the quadrant
    # This guarantees symmetry: cell[i][j] == cell[31-i][j] == cell[i][31-j] == cell[31-i][31-j]
    # And since the quadrant is symmetric (feature is centered), cell[i][j] == cell[j][i]
    cell = [[0 for _ in range(size)] for _ in range(size)]
    
    for i in range(half):
        for j in range(half):
            val = quadrant[i][j]
            cell[i][j] = val
            cell[i][size - 1 - j] = val
            cell[size - 1 - i][j] = val
            cell[size - 1 - i][size - 1 - j] = val
            
    return cell
# EVOLVE-BLOCK-END
