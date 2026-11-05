"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 17
    # Quadratic residues modulo 17
    residues = {1, 2, 4, 8, 9, 13, 15, 16}
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            if d in residues:
                colors.append(0)
            else:
                colors.append(1)
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
