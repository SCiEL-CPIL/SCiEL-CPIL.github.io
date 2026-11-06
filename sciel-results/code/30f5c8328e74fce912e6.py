"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def priority(vector, n):
    """
    Assigns a pseudo-random priority to each vector.
    
    The seed program used a uniform priority (0.0), resulting in lexicographic
    greedy insertion. Lexicographic order is suboptimal for finding large cap sets.
    
    By using a deterministic pseudo-random hash of the vector, we randomize the
    insertion order. Randomized greedy construction is a standard heuristic that
    typically outperforms structured (lexicographic) orders for packing problems
    like cap sets in small dimensions.
    """
    # Use a simple, fast hash function to generate a pseudo-random priority.
    # We ensure the result is a float for the return interface.
    # The hash should mix the bits of the vector well.
    h = 5381
    for v in vector:
        # djb2 hash variant
        h = ((h * 33) + v) & 0xFFFFFFFF
    return float(h)
# EVOLVE-BLOCK-END
