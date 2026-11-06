"""
Program to construct a 2-coloring of K_n without a monochromatic K_{2,2,2}.
Uses the Paley graph construction for n=29, which is known to be K_{2,2,2}-free
in both the graph and its complement.
"""

def construct():
    """
    Returns a dictionary with 'n' and 'colors'.
    'colors' is a list of 0s and 1s for each pair (i, j) with i < j in lexicographic order.
    """
    # The known maximum n for R(K_{2,2,2}, K_{2,2,2}) > n is 29.
    n = 29
    q = 29
    
    # Calculate quadratic residues mod q
    residues = set()
    for x in range(1, q):
        residues.add((x * x) % q)
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % q
            # Color 0 if diff is a quadratic residue, 1 otherwise
            if diff in residues:
                colors.append(0)
            else:
                colors.append(1)
                
    return {'n': n, 'colors': colors}

if __name__ == "__main__":
    res = construct()
    print(f"n={res['n']}, len(colors)={len(res['colors'])}")
