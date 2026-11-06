import time
from math import gcd

def construct():
    deadline = time.time() + 85
    n = 255
    
    # Candidates: numbers in [1,255] with gcd(v,255)>1
    # i.e., divisible by 3, 5, or 17
    candidates = [v for v in range(1, n+1) if v % 3 == 0 or v % 5 == 0 or v % 17 == 0]
    
    # Build adjacency
    m = len(candidates)
    adj = [set() for _ in range(m)]
    for i in range(m):
        for j in range(i+1, m):
            if gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Must include 255
    idx_255 = candidates.index(255)
    
    # Precompute degrees for ordering
    deg = [len(adj[i]) for i in range(m)]
    
    # Initial best: all multiples of 3
    mult3 = set(i for i in range(m) if candidates[i] % 3 == 0)
    bk_best = mult3.copy()
    bk_best_sz = len(bk_best)
    
    # Bron-Kerbosch with pivot and upper bound pruning
    import sys
    sys.setrecursionlimit(10000)
    
    def bk(R, P, X):
        nonlocal bk_best, bk_best_sz
        if time.time() > deadline - 3:
            return
        # Upper bound pruning
        if len(R) + len(P) <= bk_best_sz:
            return
        if not P and not X:
            if len(R) > bk_best_sz:
                bk_best_sz = len(R)
                bk_best = R.copy()
            return
        # Pivot
        if P:
            u = max(P, key=lambda v: len(adj[v] & P))
            Q = P - adj[u]
        else:
            Q = set()
        # Order vertices by degree for better pruning
        for v in sorted(Q, key=lambda x: -deg[x]):
            if v not in P:
                continue
            bk(R | {v}, P & adj[v], X & adj[v])
            P.discard(v)
            X.add(v)
    
    # Must include 255
    P0 = adj[idx_255].copy()
    bk({idx_255}, P0, set())
    
    result = sorted([candidates[i] for i in bk_best])
    
    # Verify
    assert 255 in result
    assert all(1 <= x <= 255 for x in result)
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            assert gcd(result[i], result[j]) > 1
    
    return result
