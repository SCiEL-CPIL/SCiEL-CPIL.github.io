"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from itertools import combinations

def construct():
    # Step 1: Generate candidate vertices
    # Candidates must share a factor with 255 (3, 5, 17)
    # and be in [1, 255]
    candidates = []
    for i in range(1, 256):
        if math.gcd(i, 255) > 1:
            candidates.append(i)
    
    n = len(candidates)
    if n == 0:
        return [255]
        
    # Step 2: Build adjacency list
    # adj[i] is a list of indices j such that gcd(candidates[i], candidates[j]) > 1
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].append(j)
                adj[j].append(i)
    
    # Step 3: Bron-Kerbosch with pivots to find maximum clique
    best_clique = []
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = R[:]
            return
        
        # Choose pivot u from P union X to maximize the number of neighbors in P
        # This reduces the number of recursive calls
        pivot = None
        max_neighbors = -1
        for u in P:
            if len([v for v in P if v in adj[u]]) > max_neighbors:
                max_neighbors = len([v for v in P if v in adj[u]])
                pivot = u
        if pivot is None:
            for u in X:
                if len([v for v in P if v in adj[u]]) > max_neighbors:
                    max_neighbors = len([v for v in P if v in adj[u]])
                    pivot = u
        
        if pivot is None:
            # No pivot, iterate over all P
            candidates_to_iterate = P
        else:
            # Iterate over P \ N(pivot)
            candidates_to_iterate = [v for v in P if v not in adj[pivot]]
            
        for v in candidates_to_iterate:
            new_R = R + [v]
            new_P = [u for u in P if u in adj[v]]
            new_X = [u for u in X if u in adj[v]]
            bron_kerbosch(new_R, new_P, new_X)
            P.remove(v)
            X.append(v)
            
    # Initialize
    # Since 255 is in candidates and is universal (connected to all in candidates),
    # we can start with R=[index_of_255] to ensure it's in the clique, 
    # or just run on all candidates. 
    # Actually, since 255 is universal, the max clique will contain it.
    # Let's just run on all candidates.
    bron_kerbosch([], candidates[:], [])
    
    # Convert indices back to values
    return [candidates[i] for i in best_clique]

# EVOLVE-BLOCK-END
