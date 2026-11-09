import time

def construct():
    """
    Constructs a 2-coloring of K_n that avoids monochromatic K_{2,2,2}.
    Uses the Paley graph of order 29, which is known to provide a lower bound
    R(K_{2,2,2}) > 29.
    """
    n = 29
    
    # Generate quadratic residues modulo 29
    # A number x is a quadratic residue mod p if x^2 ≡ a (mod p) has a solution.
    # For prime p, the non-zero quadratic residues are {x^2 % p for x in 1..p-1}
    squares = set()
    for x in range(1, n):
        squares.add((x * x) % n)
    
    # The Paley graph is defined such that i is connected to j if (j-i) mod p is a quadratic residue.
    # Since p=29 ≡ 1 (mod 4), -1 is a quadratic residue, so the graph is undirected (symmetric).
    # We color edge (i, j) with 0 if (j-i) % n is in squares, else 1.
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in squares:
                colors.append(0)
            else:
                colors.append(1)
                
    return {'n': n, 'colors': colors}
