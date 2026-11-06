import itertools

def construct():
    p = 101
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, p)
        
    denominators = list(range(1, 11))
    
    # Track the minimum size subset found for each residue
    min_size = [float('inf')] * p
    best_comb = [None] * p
    
    for k in range(1, 11):
        for comb in itertools.combinations(denominators, k):
            s = sum(inv[x] for x in comb) % p
            if k < min_size[s]:
                min_size[s] = k
                best_comb[s] = list(comb)
                
        # Early termination if all residues have been covered
        if all(x < float('inf') for x in min_size):
            break
            
    return [best_comb[r] for r in range(p)]
