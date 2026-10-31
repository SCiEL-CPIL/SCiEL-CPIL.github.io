"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def priority(vector, n):
    # Vectors with entries only in {0, 1} form a cap set of size 2^n.
    # Any vector with a 2 forms a line with two vectors in {0, 1}^n.
    # Thus, prioritizing {0, 1}^n vectors allows the greedy algorithm to
    # select the entire set {0, 1}^n, which is a valid cap set.
    if 2 in vector:
        return 0.0
    return 1.0
# EVOLVE-BLOCK-END
