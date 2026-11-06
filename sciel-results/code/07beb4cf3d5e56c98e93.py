"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Precompute modular inverses for denominators 1 to 10 modulo 101
    inverses = [0] * 11
    for x in range(1, 11):
        inverses[x] = pow(x, -1, 101)
    
    # Generate all non-empty subsets with their size, residue, and terms
    subsets = []
    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        for x in range(1, 11):
            if mask & (1 << (x - 1)):
                terms.append(x)
                s += inverses[x]
        residue = s % 101
        size = len(terms)
        subsets.append((size, residue, terms))
    
    # Sort subsets by size to ensure we pick the smallest valid subset for each residue first
    subsets.sort(key=lambda x: x[0])
    
    best_terms = [None] * 101
    covered = 0
    
    for size, residue, terms in subsets:
        if best_terms[residue] is None:
            best_terms[residue] = terms
            covered += 1
            if covered == 101:
                break
                
    # Fallback for any missing residues (should not happen)
    for r in range(101):
        if best_terms[r] is None:
            # Find any subset that sums to r
            for size, res, terms in subsets:
                if res == r:
                    best_terms[r] = terms
                    break
                    
    return best_terms
# EVOLVE-BLOCK-END
