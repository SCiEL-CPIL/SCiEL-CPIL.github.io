import time
from math import gcd
import sys

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
    
    # Find max clique containing 255
    idx_255 = candidates.index(255)
    
    # Start with all multiples of 3 as initial best
    mult3 = set(i for i in range(m) if candidates[i] % 3 == 0)
    best = mult3.copy()
    best_sz = len(best)
    
    sys.setrecursionlimit(10000)
    
    # Use Bron-Kerbosch with pivot for maximum clique
    def bk(R, P, X):
        nonlocal best, best_sz
        if time.time() > deadline - 2:
            return
        if not P and not X:
            if len(R) > best_sz:
                best_sz = len(R)
                best = R.copy()
            return
        
        # Pivot selection: choose vertex in P with max degree in P
        if P:
            u = max(P, key=lambda v: len(adj[v] & P))
            Q = P - adj[u]
        else:
            Q = set()
        
        for v in list(Q):
            if v not in P:
                continue
            bk(R | {v}, P & adj[v], X & adj[v])
            P.discard(v)
            X.add(v)
    
    # Must include 255
    P0 = adj[idx_255].copy()
    bk({idx_255}, P0, set())
    
    result = sorted([candidates[i] for i in best])
    
    # Verify
    assert 255 in result
    assert all(1 <= x <= 255 for x in result)
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            assert gcd(result[i], result[j]) > 1
    
    return result
