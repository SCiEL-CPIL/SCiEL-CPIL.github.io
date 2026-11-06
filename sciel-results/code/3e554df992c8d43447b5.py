"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    """
    Find the largest subset S of [1, 512] such that lcm(a,b) <= 512 for all a,b in S.
    Uses Bron-Kerbosch with pivot to find maximum clique.
    """
    start_time = time.time()
    deadline = start_time + 85
    
    N = 512
    
    # Build adjacency as sets for fast intersection
    adj = [set() for _ in range(N + 1)]
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = gcd(i, j)
            if i * j // g <= N:
                adj[i].add(j)
                adj[j].add(i)
    
    best_clique = []
    
    def bk(R, P, X):
        nonlocal best_clique
        
        if time.time() > deadline:
            return
        
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        
        if not P:
            return
        
        # Choose pivot from P ∪ X that maximizes |P ∩ adj[u]|
        PX = P | X
        pivot = max(PX, key=lambda u: len(P & adj[u]))
        
        # Iterate over P \ N(pivot)
        candidates = P - adj[pivot]
        for v in list(candidates):
            bk(R | {v}, P & adj[v], X & adj[v])
            P.discard(v)
            X.add(v)
    
    all_nodes = set(range(1, N + 1))
    bk(set(), all_nodes, set())
    
    if not best_clique:
        from math import isqrt
        return list(range(1, isqrt(N) + 1))
    
    return best_clique

# EVOLVE-BLOCK-END
