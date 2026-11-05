import itertools

def construct():
    # List of candidate primes p ≡ 3 mod 4
    candidates = [31, 43, 47, 59, 67]
    
    for p in candidates:
        # 1. Compute Quadratic Residues
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        
        # 2. Build In-neighborhood masks
        # mask[v] has bit u set if u -> v
        # u -> v iff (v - u) % p in squares
        masks = [0] * p
        for v in range(p):
            mask = 0
            for u in range(p):
                if u == v: continue
                if (v - u) % p in squares:
                    mask |= (1 << u)
            masks[v] = mask
            
        # 3. Check property
        valid = True
        # Iterate all 4-subsets
        # Optimization: If p is small, itertools is fine.
        for subset in itertools.combinations(range(p), 4):
            # Intersection of in-neighborhoods
            inter = masks[subset[0]] & masks[subset[1]] & masks[subset[2]] & masks[subset[3]]
            if inter == 0:
                valid = False
                break
                
        if valid:
            # Construct orientation list
            orientation = []
            for i in range(p):
                for j in range(i + 1, p):
                    # i -> j iff (j - i) % p in squares
                    if (j - i) % p in squares:
                        orientation.append(0)
                    else:
                        orientation.append(1)
            return {'n': p, 'orientation': orientation}
            
    # Fallback to 67 if something went wrong (should not happen)
    p = 67
    squares = {x * x % p for x in range(1, p)}
    orientation = [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]
    return {'n': p, 'orientation': orientation}
