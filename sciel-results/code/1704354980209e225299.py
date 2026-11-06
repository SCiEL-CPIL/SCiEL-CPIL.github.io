"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def priority(vector, n):
    # Prioritize vectors with fewer 1s.
    # Vectors in {0, 2}^n (0 ones) form a cap set of size 2^n.
    # By adding them first, we maximize the base size.
    count_ones = 0
    for v in vector:
        if v == 1:
            count_ones += 1
    return -count_ones
# EVOLVE-BLOCK-END
