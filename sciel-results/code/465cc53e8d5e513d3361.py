"""OneMax direct binary construction (100 bits)."""
N = 100

# EVOLVE-BLOCK-START
def construct():
    # For the OneMax objective on a 100-bit binary string, the global optimum
    # is the all-ones string, which yields a score of 100 (the maximum possible).
    # The seed program returned "01" * 50, yielding 50 ones (score 50), giving
    # a search score of 50/50 = 1.0. Returning "1" * N yields 100 ones,
    # giving a search score of 100/50 = 2.0.
    return "1" * N
# EVOLVE-BLOCK-END
