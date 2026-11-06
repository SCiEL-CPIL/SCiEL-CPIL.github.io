import math
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    candidates = [i for i in range(1, 256) if math.gcd(i, 255) > 1]
    if 255 not in candidates:
        candidates.append(255)
    
    n = len(candidates)
    idx = {v: i for i, v in enumerate(candidates)}
    idx_255 = idx[255]
    
    # Build adjacency as bitmasks
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Calculate degrees
    degrees = [0] * n
    for i in range(n):
        degrees[i] = bin(adj[i]).count("1")
    
    best_size = 1
    best_mask = (1 << idx_255)
    
    def popcount(x):
        return bin(x).count("1")
    
    # Greedy coloring for upper bound
    def color_bound(P):
        if P == 0:
            return 0
        color_class = 0
        remaining = P
        while remaining:
            color_class += 1
            independent = 0
            temp = remaining
            while temp:
                lb = temp & (-temp)
                v = lb.bit_length() - 1
                if (independent & adj[v]) == 0:
                    independent |= lb
                temp ^= lb
            remaining &= ~independent
        return color_class
    
    def bron_kerbosch(R, P, X):
        nonlocal best_size, best_mask
        if time.time() - start_time > DEADLINE:
            return
        
        r_size = popcount(R)
        
        # Upper bound pruning
        if r_size + color_bound(P) <= best_size:
            return
        
        if P == 0 and X == 0:
            if r_size > best_size:
                best_size = r_size
                best_mask = R
            return
        
        if P == 0:
            return
        
        # Pivot: choose from P|X with max degree in P
        U = P | X
        if U == 0:
            return
        pivot = -1
        max_deg = -1
        temp = U
        while temp:
            lb = temp & (-temp)
            v = lb.bit_length() - 1
            deg = popcount(P & adj[v])
            if deg > max_deg:
                max_deg = deg
                pivot = v
            temp ^= lb
        
        # Candidates: P vertices not adjacent to pivot
        cand = P & ~adj[pivot]
        
        while cand:
            if time.time() - start_time > DEADLINE:
                return
            lb = cand & (-cand)
            v = lb.bit_length() - 1
            bron_kerbosch(R | lb, P & adj[v], X & adj[v])
            P ^= lb
            X |= lb
            cand ^= lb
    
    # Start with a large initial clique: all multiples of 3
    mult3 = [i for i in range(1, 256) if i % 3 == 0]
    init_mask = 0
    for v in mult3:
        if v in idx:
            init_mask |= (1 << idx[v])
    
    init_size = popcount(init_mask)
    best_size = max(best_size, init_size)
    best_mask = init_mask
    
    # Find vertices not in init_mask adjacent to ALL in init_mask
    P = 0
    for i in range(n):
        if not (init_mask & (1 << i)):
            if (init_mask & ~adj[i]) == 0:
                P |= (1 << i)
    
    bron_kerbosch(init_mask, P, 0)
    
    # Also try from 255
    R = (1 << idx_255)
    P2 = adj[idx_255]
    X2 = 0
    bron_kerbosch(R, P2, X2)
    
    result = []
    m = best_mask
    while m:
        lb = m & (-m)
        v = lb.bit_length() - 1
        result.append(candidates[v])
        m ^= lb
    result.sort()
    
    if 255 not in result:
        result.append(255)
        result.sort()
    
    return result
