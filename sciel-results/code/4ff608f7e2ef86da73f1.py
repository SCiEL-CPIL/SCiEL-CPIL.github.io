"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def design(problem):
    """A centred air hole of radius 0.42 lattice constants in a dielectric background."""
    size = problem["pixels"]
    cell = []
    # Using a hole (air in dielectric) instead of a rod (dielectric in air)
    # Radius tuned for high-contrast (eps=12) square lattice TM band gap
    r = 0.42
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        row = []
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            # Inside the hole is air (0), outside is dielectric (1)
            if x * x + y * y < r * r:
                row.append(0)
            else:
                row.append(1)
        cell.append(row)
    return cell
# EVOLVE-BLOCK-END
