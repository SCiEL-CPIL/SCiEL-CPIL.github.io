from fractions import Fraction

def construct():
    # Candidate set: {5, 6, 7, 9, 10, ..., 24}
    # Missing 8 to break 1/6 = 1/8 + 1/24
    # Check if valid
    S = [5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    
    # Verify validity
    Sums = {Fraction(0, 1)}
    valid = True
    for x in S:
        rx = Fraction(1, x)
        if rx in Sums:
            valid = False
            break
        # Check if adding x makes any existing a invalid
        # This part is skipped in the hardcoded check for efficiency if we assume the set is constructed carefully,
        # but we should verify the full condition.
        # The condition is: for all a in S, 1/a is not in Sums(S \ {a}).
        # My loop above checks if 1/x is in Sums of PREVIOUS elements.
        # This only checks one direction. We need to check if 1/a (existing) becomes a sum of NEW set.
        # i.e., 1/a = rx + sum(T) where T subset of previous S.
        # sum(T) = 1/a - rx.
        # This requires checking all pairs.
        
        # For the hardcoded set, let's do a full check.
        pass

    # Full check for the hardcoded set
    S_set = set(S)
    # Generate all subset sums? 2^19 is too big.
    # But we can check each element a against sums of others.
    # For each a, check if 1/a is in subset sums of S \ {a}.
    
    # This is expensive. Let's rely on the greedy program's logic but with a better order or specific set.
    
    # Actually, the greedy program from 24 down to 1 produced {7..24}.
    # If I run greedy from 24 down to 1, but SKIP 8, I might get {5, 6, 7, 9..24}.
    
    # Let's try a modified greedy:
    # Iterate 24 down to 1.
    # If x == 8, skip.
    # Else, try to add.
    
    S = []
    Sums = {Fraction(0, 1)}
    
    for x in range(24, 0, -1):
        if x == 8:
            continue
            
        rx = Fraction(1, x)
        
        # Check if rx is in Sums
        if rx in Sums:
            continue
            
        valid_add = True
        for a in S:
            ra = Fraction(1, a)
            target = ra - rx
            if target > 0:
                if target in Sums:
                    valid_add = False
                    break
                    
        if valid_add:
            S.append(x)
            new_sums = set(Sums)
            for s in Sums:
                new_sums.add(s + rx)
            Sums = new_sums
            
    return S
