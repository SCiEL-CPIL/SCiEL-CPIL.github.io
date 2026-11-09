import time

def construct():
    MOD = 101
    N = 10
    
    # Precompute modular inverses
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, -1, MOD)
    
    # For each residue, store the best (smallest) subset
    best_len = [float('inf')] * MOD
    best_subset = [None] * MOD
    
    # Enumerate all non-empty subsets of {1,...,10}
    for mask in range(1, 1 << N):
        s = 0
        count = 0
        subset = []
        for i in range(N):
            if mask & (1 << i):
                d = i + 1
                s = (s + inv[d]) % MOD
                count += 1
                subset.append(d)
        
        if count < best_len[s]:
            best_len[s] = count
            best_subset[s] = subset
    
    # Build result
    result = []
    for r in range(MOD):
        result.append(best_subset[r])
    
    return result
