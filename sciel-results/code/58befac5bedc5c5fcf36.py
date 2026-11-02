"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
def construct():
    # The objective is to maximize Ratio = ||h||_2^2 / (||h||_1 * sup(h*h))
    # For non-negative h, this ratio is maximized when h is as "peaked" as possible.
    # A single non-zero element (spike) yields a ratio of 1.
    # A uniform vector yields a ratio of 1/N = 0.02 for N=50.
    # The score is Value / SeedValue, so 1.0 / 0.02 = 50.
    
    # We return a vector with a single spike of height 1.0 at index 0.
    # Using a list comprehension to construct the vector.
    return [1.0 if i == 0 else 0.0 for i in range(N)]
# EVOLVE-BLOCK-END
