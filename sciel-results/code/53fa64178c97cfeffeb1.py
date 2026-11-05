import time

def construct():
    # The Ramsey number R(K_{2,2,2}) is known to be 30.
    # This means that any 2-coloring of the edges of K_{30} contains a monochromatic K_{2,2,2}.
    # Therefore, the maximum n for which a valid 2-coloring exists is 29.
    # The Paley graph of order 29 provides such a coloring.
    
    n = 29
    # Paley graph construction:
    # Vertices are 0..28.
    # Edge (i, j) is color 0 if (j-i) is a quadratic residue mod 29, else color 1.
    # Note: The choice of which color is 0 vs 1 doesn't matter for validity, 
    # as long as it's consistent.
    
    # Calculate quadratic residues mod 29
    residues = set()
    for x in range(1, n):
        residues.add((x * x) % n)
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in residues:
                colors.append(0)
            else:
                colors.append(1)
                
    return {'n': n, 'colors': colors}
