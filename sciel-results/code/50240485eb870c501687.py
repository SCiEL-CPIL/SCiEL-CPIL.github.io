import time

def construct():
    start_time = time.time()
    LIMIT = 1009
    MAX_VAL = 1008
    
    # Precompute modular inverses
    inv = [0] * (LIMIT + 1)
    for i in range(1, LIMIT):
        inv[i] = pow(i, -1, LIMIT)
    
    # Build pairs_by_a
    # pairs_by_a[a] contains b such that (a, b) is a conflict pair with a < b
    # We only consider a, b in [2, 1007] because 1 and 1008 are self-inverses and forbidden.
    pairs_by_a = [[] for _ in range(LIMIT + 1)]
    
    for i in range(2, 1008):
        j = inv[i]
        # Since i in [2, 1007], j is also in [2, 1007] and j != i
        a, b = min(i, j), max(i, j)
        pairs_by_a[a].append(b)
        
    # Compute min_b_limit[s]
    # min_b_limit[s] = min { b | (a, b) in pairs, a >= s }
    # If no such pair, min_b_limit[s] = 1008 (infinity)
    min_b_limit = [1008] * (LIMIT + 1)
    
    current_min_b = 1008
    for s in range(1008, 0, -1):
        for b in pairs_by_a[s]:
            if b < current_min_b:
                current_min_b = b
        min_b_limit[s] = current_min_b
        
    best_len = 0
    best_start = 2
    
    # Search for best start
    # Valid starts are in [2, 1007]
    for s in range(2, 1008):
        # Check time
        if time.time() - start_time > 80:
            break
            
        max_e = min_b_limit[s] - 1
        # Clamp to valid range [2, 1007]
        if max_e > 1007:
            max_e = 1007
        if max_e < s:
            continue
            
        length = max_e - s + 1
        if length > best_len:
            best_len = length
            best_start = s
            
    return [best_start, best_len]
