"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def design(problem):
    """
    A dielectric background with a central circular air hole.
    This is the dual of the seed (rod in air) and typically yields larger
    TM band gaps for high dielectric contrast materials.
    """
    size = problem["pixels"]
    cell = []
    center = (size - 1) / 2.0
    # Radius in pixels. 0.2 lattice constants was used for the rod.
    # For a hole in dielectric, a slightly larger radius is often optimal.
    # Let's try 0.25 lattice constants -> 8 pixels.
    radius = 0.25 * size 
    
    for i in range(size):
        row = []
        for j in range(size):
            # Pixel center coordinates
            x = i - center
            y = j - center
            dist_sq = x * x + y * y
            if dist_sq < radius * radius:
                row.append(0)  # Air hole
            else:
                row.append(1)  # Dielectric
        cell.append(row)
    return cell
# EVOLVE-BLOCK-END
