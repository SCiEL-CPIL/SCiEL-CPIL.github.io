"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Paley graph on 41 vertices: i ~ j when j - i is a nonzero quadratic residue mod 41.
    # This graph is (5,5)-free, proving R(5,5) > 41.
    q = 41
    residues = {(x * x) % q for x in range(1, q)}
    return q, [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
# EVOLVE-BLOCK-END
