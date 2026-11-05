"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
def construct():
    # OneMax with 100 bits: the global optimum is all ones.
    # The trivial optimal solution is "1" * N, which scores 100.
    # No search is needed; we return the optimal string directly.
    return "1" * N
# EVOLVE-BLOCK-END
