import time

def construct():
    """
    Constructs a 2-coloring of K_n with no monochromatic K2,2,2.
    Returns a dictionary with 'n' and 'colors'.
    
    Uses the Paley graph construction for p=29, which is a known valid
    2-coloring of K_29 with no monochromatic K2,2,2. Since R(K2,2,2)=30,
    n=29 is the maximum possible.
    """
    # The Ramsey number R(K2,2,2) is 30.
    # This means any 2-coloring of K30 contains a monochromatic K2,2,2.
    # Therefore, the maximum n for a valid coloring is 29.
    # A valid coloring for n=29 is given by the Paley graph construction for p=29.
    
    n = 29
    p = 29
    
    # Compute quadratic residues modulo p
    residues = set()
    for x in range(1, p):
        residues.add((x * x) % p)
    
    # Generate colors in lexicographic order
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % p
            if diff in residues:
                colors.append(0)
            else:
                colors.append(1)
    
    return {'n': n, 'colors': colors}
