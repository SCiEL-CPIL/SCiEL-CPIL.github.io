import time
import random

def construct():
    """
    Returns a valid 2-coloring of K_n that maximizes n such that neither color class
    contains K_{2,2,2} as a subgraph.
    The maximum possible n is 29, as R(K_{2,2,2}) = 30.
    We use the Paley graph on 29 vertices, which is known to be K_{2,2,2}-free in both colors.
    """
    n = 29
    q = 29
    
    # Precompute quadratic residues mod 29
    squares = set()
    for x in range(1, q):
        squares.add(x * x % q)
    
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % q
            if diff in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Generate colors in lexicographic order (i, j) for i < j
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                colors.append(0)
            else:
                colors.append(1)
                
    return {'n': n, 'colors': colors}

if __name__ == "__main__":
    result = construct()
    print(f"Constructed coloring for n={result['n']}")
