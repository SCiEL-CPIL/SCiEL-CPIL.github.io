"""Improved program: use the known Paley(29) construction for R(K2,2,2)."""

# EVOLVE-BLOCK-START
def construct():
    # Paley graph on 29 vertices (29 is a prime ≡ 1 mod 4).
    # Colour 0 (red) on pairs whose difference is a nonzero quadratic residue mod 29.
    # This is a self-complementary (symmetric) graph, so the complement (colour 1) is isomorphic.
    # It is a known candidate for a K2,2,2-free colouring of K29.
    p = 29
    squares = {x * x % p for x in range(1, p)}
    n = p
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % p
            colors.append(0 if d in squares else 1)
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
