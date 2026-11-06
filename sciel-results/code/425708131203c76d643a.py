"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def design(problem):
    """
    A square air hole in a dielectric background.
    High-index background structures typically exhibit larger TM band gaps.
    The square shape aligns with the lattice symmetry.
    Size 16x16 (0.5 lattice constants) is chosen to maximize the gap.
    """
    size = problem["pixels"]
    # Initialize with dielectric (1)
    cell = [[1 for _ in range(size)] for _ in range(size)]
    
    # Define the square air hole (0)
    hole_size = 16
    start = size // 2 - hole_size // 2
    end = start + hole_size
    
    for i in range(start, end):
        for j in range(start, end):
            cell[i][j] = 0
            
    return cell
# EVOLVE-BLOCK-END
