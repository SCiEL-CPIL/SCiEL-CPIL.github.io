"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Valid set of 7 disjoint residue classes with distinct moduli (powers of 2)
    # Residues: 1, 2, 4, 8, 16, 32, 64
    # Moduli:   2, 4, 8, 16, 32, 64, 128
    return [[1, 2], [2, 4], [4, 8], [8, 16], [16, 32], [32, 64], [64, 128]]
# EVOLVE-BLOCK-END
