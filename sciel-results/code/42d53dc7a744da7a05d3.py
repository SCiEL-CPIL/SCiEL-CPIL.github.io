"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def design(problem):
    """
    Designs a 32x32 unit cell using concentric square rings with alternating
    dielectric (1) and air (0) assignments. This high-spatial-frequency
    structure is designed to open a large gap between higher-order TM bands.
    """
    size = problem["pixels"]
    cell = [[0 for _ in range(size)] for _ in range(size)]
    
    # The center of the 32x32 grid is at (15.5, 15.5)
    center = (size - 1) / 2.0
    
    for i in range(size):
        for j in range(size):
            # Calculate Chebyshev distance from the center
            dx = abs(i - center)
            dy = abs(j - center)
            dist = max(dx, dy)
            
            # Determine the ring index (1 to 16)
            # Since center is x.5 and i,j are integers, dist is always x.5
            k = int(dist) + 1
            
            # Alternate material for each ring
            # Ring 1 (center) is Dielectric (1)
            # Ring 2 is Air (0)
            # etc.
            val = 1 if k % 2 == 1 else 0
            cell[i][j] = val
            
    return cell
# EVOLVE-BLOCK-END
