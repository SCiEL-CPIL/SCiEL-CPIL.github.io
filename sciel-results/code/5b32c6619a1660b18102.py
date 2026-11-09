import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def build_paley_orientation(p):
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                diff = (j - i) % p
                if diff in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return orientation

    def is_valid_paley_fast(p):
        """Fast validation using out-neighborhood bitmasks.
        A 4-set {i,j,k,l} has a common dominator iff 
        N[i] & N[j] & N[k] & N[l] != 0, where N[v] is the bitmask of 
        vertices that v dominates (out-neighborhood of v).
        """
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        
        # Precompute out-neighborhood masks
        # N[v] has bit u set if v -> u (i.e., v dominates u)
        N = [0] * p
        for v in range(p):
            mask = 0
            for u in range(p):
                if v == u:
                    continue
                diff = (u - v) % p
                if diff in squares:
                    mask |= (1 << u)
            N[v] = mask
        
        # Check all 4-sets: for each (i,j,k,l), check if N[i]&N[j]&N[k]&N[l] != 0
        # Early exit optimization: if N[i]&N[j]&N[k] == 0, skip all l
        for i in range(p):
            if time.time() > deadline:
                return False
            for j in range(i + 1, p):
                if time.time() > deadline:
                    return False
                common_ij = N[i] & N[j]
                if common_ij == 0:
                    continue
                for k in range(j + 1, p):
                    if time.time() > deadline:
                        return False
                    common_ijk = common_ij & N[k]
                    if common_ijk == 0:
                        continue
                    for l in range(k + 1, p):
                        if (common_ijk & N[l]) == 0:
                            return False
        return True

    # Try Paley tournaments for primes p ≡ 3 (mod 4)
    # Smaller p gives better score, so try smallest first
    candidates = [31, 43, 47, 59, 67]
    
    best_n = 67
    best_orientation = build_paley_orientation(67)
    
    for p in candidates:
        if time.time() > deadline - 5:
            break
        if is_valid_paley_fast(p):
            best_n = p
            best_orientation = build_paley_orientation(p)
            break
    
    return {'n': best_n, 'orientation': best_orientation}
