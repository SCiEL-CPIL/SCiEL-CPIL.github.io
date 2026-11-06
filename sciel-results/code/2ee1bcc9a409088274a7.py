def _verify_paley_exact(p, orientation):
    """
    Exact verification of the 4-dominator property.
    """
    in_bits = [0] * p
    idx = 0
    for i in range(p):
        for j in range(i + 1, p):
            bit = orientation[idx]
            idx += 1
            if bit == 0:
                in_bits[j] |= (1 << i)
            else:
                in_bits[i] |= (1 << j)
    
    # Precompute pairwise intersections
    pair_inter = [[0] * p for _ in range(p)]
    for i in range(p):
        for j in range(i + 1, p):
            inter = in_bits[i] & in_bits[j]
            pair_inter[i][j] = inter
            pair_inter[j][i] = inter
            
    # Check triples first for early exit
    for a, b, c in itertools.combinations(range(p), 3):
        if (pair_inter[a][b] & in_bits[c]) == 0:
            return False
            
    # Check all 4-combinations
    for a, b, c, d in itertools.combinations(range(p), 4):
        if (pair_inter[a][b] & pair_inter[c][d]) == 0:
            return False
            
    return True
