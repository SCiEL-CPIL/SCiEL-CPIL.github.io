import itertools

def construct():
    # Points are 0..10 and 11 (representing infinity)
    # We use 11 for infinity to keep everything as integers in a list
    
    def mod_inv(x, p=11):
        return pow(x, p-2, p)

    def apply_matrix(M, x):
        a, b, c, d = M
        # x is 0..10 or 11 (infinity)
        if x == 11:
            if c == 0:
                return 11
            else:
                return (a * mod_inv(c, 11)) % 11
        else:
            if c * x + d == 0:
                return 11
            else:
                return (a * x + b) * mod_inv(c * x + d, 11) % 11

    # Generate PSL(2, 11) elements
    # Matrix [[a, b], [c, d]] with ad - bc = 1 mod 11
    # We identify M and -M as the same in PSL, so we can just generate SL and divide by 2?
    # Actually, the action of M and -M is identical.
    # So we can just generate all SL(2,11) matrices and apply them. The orbit will be the same.
    # There are 1320 matrices in SL(2,11).
    
    group_elements = []
    for a in range(11):
        for b in range(11):
            for c in range(11):
                for d in range(11):
                    if (a * d - b * c) % 11 == 1:
                        group_elements.append((a, b, c, d))
                        
    # There are 1320 elements.
    
    # Candidate base blocks
    # We try the first few lexicographic 6-subsets of 0..11
    candidates = []
    for combo in itertools.combinations(range(12), 6):
        candidates.append(combo)
        if len(candidates) >= 20:
            break
            
    for base in candidates:
        base_set = set(base)
        orbit = set()
        valid = True
        seen_5subsets = set()
        
        for M in group_elements:
            # Apply M to all points in base
            new_block = tuple(sorted(apply_matrix(M, x) for x in base_set))
            if new_block in orbit:
                continue
            orbit.add(new_block)
            
            # Check 5-subsets
            # If we find a collision, this candidate is not a block of the Steiner system
            # (or rather, the orbit is not a Steiner system)
            for sub in itertools.combinations(new_block, 5):
                if sub in seen_5subsets:
                    valid = False
                    break
                seen_5subsets.add(sub)
            if not valid:
                break
        
        if valid and len(orbit) == 132:
            # Convert to list of lists for return
            return [list(b) for b in orbit]
            
    # Fallback: Greedy (should not be reached)
    covered, blocks = set(), []
    for b in itertools.combinations(range(12), 6):
        subsets = list(itertools.combinations(b, 5))
        if not any(s in covered for s in subsets):
            blocks.append(list(b))
            covered.update(subsets)
    return blocks
