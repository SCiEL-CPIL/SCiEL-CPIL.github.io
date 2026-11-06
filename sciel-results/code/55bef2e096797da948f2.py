"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 29
    # Compute quadratic residues modulo 29
    # q = 29 is a prime, 29 % 4 == 1
    residues = set()
    for x in range(1, n):
        residues.add((x * x) % n)
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            # Color 0 if difference is a quadratic residue, else 1
            if diff in residues:
                colors.append(0)
            else:
                colors.append(1)
                
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
