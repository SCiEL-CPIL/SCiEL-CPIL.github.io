import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    N = 512
    
    # Build the compatibility graph
    # adj[i] is a set of j such that lcm(i, j) <= N
    # We use 1-based indexing for values, 0-based for list indices
    # Let's use a list of lists or sets for neighbors
    adj = [set() for _ in range(N + 1)]
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = math.gcd(i, j)
            l = (i // g) * j
            if l <= N:
                adj[i].add(j)
                adj[j].add(i)
    
    # We want to find the Maximum Clique in this graph.
    # A clique in H is an independent set in the conflict graph.
    
    # Simple Bron-Kerbosch with pivot
    # R: current clique
    # P: candidates
    # X: excluded (already processed)
    
    best_clique = []
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        
        # Choose pivot u from P union X
        # u should maximize |P \cap N(u)|
        # For simplicity and speed, we can just pick any or a greedy one.
        # A good heuristic: pick u with max degree in P.
        
        if P or X:
            # Pick pivot u
            # To optimize, we want u that maximizes |P & adj[u]|
            # This minimizes the number of branches: iterate over P \ N(u)
            
            # If P is empty, we can't pick pivot from P.
            # Standard BK uses P union X.
            # If P is empty, we are done (handled above).
            
            # Let's pick u from P union X
            candidates = list(P)
            if X:
                candidates += list(X)
            
            # Find u with max intersection with P
            max_count = -1
            u = None
            for v in candidates:
                count = len(P & adj[v])
                if count > max_count:
                    max_count = count
                    u = v
            
            # Iterate over P \ N(u)
            # Nodes in P that are NOT neighbors of u
            # If u is in P, N(u) contains neighbors.
            # We want to branch on nodes that are not neighbors of u?
            # Wait, the property is: any maximal clique must contain u or be in P \ N(u).
            # So we branch on P \ N(u).
            
            P_not_Nu = P - adj[u] if u is not None else P
            
            for v in list(P_not_Nu):
                # Recurse
                bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
                
                P.remove(v)
                X.add(v)
                
                if time.time() > deadline:
                    return

    # Initial call
    # P is all nodes 1..N
    P = set(range(1, N + 1))
    X = set()
    R = set()
    
    # To improve performance, we can sort nodes by degree descending?
    # BK is sensitive to ordering.
    
    bron_kerbosch(R, P, X)
    
    # Fallback if BK didn't finish or found nothing (shouldn't happen, single node is always clique)
    if not best_clique:
        best_clique = [1]
        
    return best_clique
