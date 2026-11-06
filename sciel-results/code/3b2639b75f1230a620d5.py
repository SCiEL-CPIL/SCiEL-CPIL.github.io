"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def priority(vector, n):
    # Heuristic: prefer vectors with fewer 1s (less likely to be middle of a 3-AP)
    ones = sum(1 for x in vector if x == 1)
    # Small hash-based perturbation for tie-breaking
    h = hash(vector) & 0xFFFFF
    return -float(ones) + h * 1e-7
# EVOLVE-BLOCK-END
