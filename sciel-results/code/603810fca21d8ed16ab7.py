import itertools

def construct():
    p = 101
    denominators = list(range(1, 11))
    n = 10

    # Precompute modular inverses
    inverses = {x: pow(x, -1, p) for x in denominators}

    # best[r] = minimum size that achieves residue r
    # combo[r] = the combination achieving it
    best_size = [n + 1] * p
    best_combo = [None] * p

    # We iterate by increasing size. 
    # As soon as all residues are covered, we can stop because 
    # we are looking for the minimum size for each residue, 
    # and any larger size would not improve the "minimum size" for 
    # already covered residues, nor would it be needed for the 
    # "minimize max row length" objective if the max is already determined 
    # by the largest size used in the current cover.
    # Actually, if we stop at size K, the max row length is K.
    # If we continued, we might find a residue that was NOT covered 
    # by size < K, but we already covered ALL residues.
    # So the max length is exactly K (or less, if some were covered earlier, 
    # but the max is K because at least one residue required size K to be 
    # covered? No, if all were covered by size < K, we would have stopped earlier).
    
    for size in range(1, n + 1):
        for combo in itertools.combinations(denominators, size):
            s = 0
            for d in combo:
                s += inverses[d]
            residue = s % p
            if best_size[residue] > size:
                best_size[residue] = size
                best_combo[residue] = list(combo)
        
        # Check if all residues are covered
        # Optimization: count covered residues
        # Since p=101 is small, a simple check is fast enough
        if all(c is not None for c in best_combo):
            break

    # Construct result
    result = []
    for r in range(p):
        if best_combo[r] is not None:
            result.append(best_combo[r])
        else:
            # Fallback (shouldn't happen if size goes up to 10)
            result.append(denominators[:])

    return result
