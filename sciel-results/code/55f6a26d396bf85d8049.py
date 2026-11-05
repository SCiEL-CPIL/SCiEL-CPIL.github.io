import numpy as np

def construct():
    """
    Returns a 2-coloring of K_n that avoids monochromatic K_{2,2,2} subgraphs.
    The maximum such n is 29, achieved by the Paley graph of order 29.
    """
    n = 29
    
    # Construct the Paley graph of order 29.
    # Vertices are 0..28.
    # Edge (i, j) is color 0 if (j-i) mod 29 is a quadratic residue, else 1.
    
    # Calculate quadratic residues mod 29
    # x^2 mod 29 for x in 1..28
    residues = set((x * x) % n for x in range(1, n))
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in residues:
                colors.append(0)
            else:
                colors.append(1)
                
    return {'n': n, 'colors': colors}
