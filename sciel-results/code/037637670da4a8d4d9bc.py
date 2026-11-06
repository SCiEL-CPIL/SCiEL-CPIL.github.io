import time
from math import gcd

def construct():
    deadline = time.time() + 82
    n = 255
    
    # Candidates: numbers in [1,255] with gcd(v,255)>1
    candidates = [v for v in range(1, n + 1) if v % 3 == 0 or v % 5 == 0 or v % 17 == 0]
    m = len(candidates)
    
    # Build adjacency as bitmasks
    adj_mask = [0] * m
    for i in range(m):
        for j in range(i + 1, m):
            if gcd(candidates[i], candidates[j]) > 1:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    
    idx_255 = candidates.index(255)
    
    # Initial best: all multiples of 3 (85 elements)
    mult3_indices = [i for i in range(m) if candidates[i] % 3 == 0]
    best_clique = set(mult3_indices)
    best_size = len(best_clique)
    
    import sys
    sys.setrecursionlimit(10000)
    
    def expand(R, P, X):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
        
        if len(R) + len(P) <= best_size:
            return
        
        if not P:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = R.copy()
            return
        
        # Pivot: choose u in P∪X maximizing |P ∩ N(u)|
        PX = P | X
        if PX:
            best_u = max(PX, key=lambda u: bin(adj_mask[u] & 0xFFFFFFFFFFFFFFFF & 
                       sum(1 << i for i in P)).count('1'))
            # Vertices to try: P - N(u)
            to_try = P - set(i for i in P if (adj_mask[best_u] >> i) & 1)
        else:
            to_try = P
        
        for v in list(to_try):
            if v not in P:
                continue
            
            new_R = R | {v}
            new_P = P & set(i for i in P if (adj_mask[v] >> i) & 1)
            new_X = X & set(i for i in X if (adj_mask[v] >> i) & 1)
            
            if len(new_R) + len(new_P) > best_size:
                expand(new_R, new_P, new_X)
            
            P.discard(v)
            X.add(v)
            
            if time.time() > deadline:
                return
    
    P0 = set(i for i in range(m) if (adj_mask[idx_255] >> i) & 1)
    expand({idx_255}, P0, set())
    
    result = sorted([candidates[i] for i in best_clique])
    
    assert 255 in result
    assert all(1 <= x <= 255 for x in result)
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            assert gcd(result[i], result[j]) > 1
    
    return result
