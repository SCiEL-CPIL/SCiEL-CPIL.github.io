import math
from itertools import combinations

def construct():
    # 1. Generate candidates
    candidates = [x for x in range(1, 256) if math.gcd(x, 255) > 1]
    n = len(candidates)
    # Map value to index for easier handling? Or just use values.
    # Using indices is faster for bitsets, but sets of values are easier to code.
    # Let's use indices.
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # 2. Build adjacency matrix/list for G_comp (edge if gcd > 1)
    # adj[i] is a set of indices j such that gcd(candidates[i], candidates[j]) > 1
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # 3. Bron-Kerbosch for Max Clique
    best_clique = set()
    best_size = 0
    
    def bk(R, P, X):
        nonlocal best_clique, best_size
        if not P and not X:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = set(R)
            return
        
        if not P:
            return

        # Pivot selection
        # Choose u in P ∪ X that maximizes |N(u) ∩ P|
        # N(u) in G_comp is adj[u]
        pivot_candidates = P.union(X)
        pivot = None
        max_count = -1
        for u in pivot_candidates:
            # Count neighbors in P
            # adj[u] contains all neighbors in G_comp
            # We want intersection with P
            count = len(adj[u] & P)
            if count > max_count:
                max_count = count
                pivot = u
        
        # If pivot is in P, we only iterate over P \ N(pivot)
        # If pivot is in X, we iterate over all P
        # Standard optimization: 
        # If pivot is in P, candidates = P - N(pivot)
        # Else candidates = P
        # Note: N(pivot) includes pivot itself? No, simple graph, no self loops.
        # adj[pivot] does not contain pivot.
        
        if pivot in P:
            candidates = P - adj[pivot]
        else:
            candidates = P
            
        for v in list(candidates): # list() to avoid modification during iteration
            # Include v in R
            new_R = R | {v}
            # Update P and X
            new_P = P & adj[v]
            new_X = X & adj[v]
            
            bk(new_R, new_P, new_X)
            
            # Move v from P to X
            P.remove(v)
            X.add(v)
            
    # Initial call
    # R = empty, P = all nodes, X = empty
    bk(set(), set(range(n)), set())
    
    # Map indices back to values
    result = [candidates[i] for i in best_clique]
    
    # Ensure 255 is in result. 
    # Since 255 is connected to all candidates (gcd(255, x) > 1 for all x in candidates),
    # it is a universal vertex in G_comp. 
    # Any maximal clique will include universal vertices? 
    # Actually, if a vertex is universal, it can be added to any clique.
    # So the maximum clique must contain 255.
    # Let's verify.
    if 255 not in result:
        # This shouldn't happen, but just in case
        # Add it? No, we must return a valid set.
        # If BK missed it, something is wrong.
        # But 255 is connected to everyone in candidates.
        # So any clique C can be extended to C U {255} if 255 not in C.
        # Thus Max Clique MUST contain 255.
        pass
        
    return result
