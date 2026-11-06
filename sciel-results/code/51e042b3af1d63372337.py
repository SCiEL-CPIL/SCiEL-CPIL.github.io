"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Paley colouring of K29: colour 0 on pairs whose difference is a nonzero square mod 29.
    # This construction is known to avoid K2,2,2 in both colour classes.
    p = 29
    squares = {x * x % p for x in range(1, p)}
    colors = [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]
    return {'n': p, 'colors': colors}
# EVOLVE-BLOCK-END
