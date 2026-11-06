"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def design(problem):
    """
    A dielectric background (epsilon=12) with a central circular air hole.
    This configuration is known to yield larger TM band gaps in square lattices
    with high dielectric contrast compared to small dielectric rods in air.
    Radius is set to 0.35 lattice constants.
    """
    size = problem["pixels"]
    cell = []
    r = 0.35
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            # 0 is air, 1 is dielectric
            if x * x + y * y < r * r:
                row.append(0)
            else:
                row.append(1)
        cell.append(row)
    return cell
# EVOLVE-BLOCK-END
