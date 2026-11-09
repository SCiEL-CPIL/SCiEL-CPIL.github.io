import time
import itertools

def construct():
    """
    Return a valid 2-coloring of K_n (n=29) with no monochromatic K2,2,2.
    Uses the Paley graph on 29 vertices (prime, 1 mod 4), which is known to be
    a Ramsey (K2,2,2)-free coloring.
    """
    # The Paley graph on p vertices (p prime, p ≡ 1 mod 4) gives a valid coloring
    # for R(K2,2,2) lower bound. For p=29, this achieves n=29.
    n = 29
    
    # Quadratic residues mod 29
    squares = set()
    for x in range(1, n):
        squares.add((x * x) % n)
    
    # Build colors in lexicographic order (i, j) with i < j
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in squares:
                colors.append(0)
            else:
                colors.append(1)
    
    return {'n': n, 'colors': colors}
