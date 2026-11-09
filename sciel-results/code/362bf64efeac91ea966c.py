"""OneMax direct binary construction (100 bits)."""
N = 100

# EVOLVE-BLOCK-START
def construct():
    # Build the all-ones genotype via a comprehension over a range of length N.
    # This yields a string of exactly N '1' characters, the global optimum.
    return ''.join('1' for _ in range(N))
# EVOLVE-BLOCK-END
